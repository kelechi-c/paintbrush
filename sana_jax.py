import jax, math
from flax import nnx
from jax import Array, numpy as jnp
from typing import Optional, Callable

class play_config:
    

class sana_config:
    name = "SanaTransformer2DModel"
    channels = 32
    dim = 1152
    cross_attn_dim = 1152
    cross_attn_heads = 16
    attn_heads = 36
    attn_head_dim = 32
    caption_channels = 2304
    num_layers = 28
    patch_size = 1
    sample_size = 32
    attention_bias = False
    cross_attn_head_dim = 72
    dropout = 0.0
    mlp_ratio = 2.5
    norm_elementwise_affine = False
    norm_eps = 1e-06
    out_channels = 32


rngs = nnx.Rngs(0)

class GlumbConv(nnx.Module):
    def __init__(self, in_ch, out_ch, ratio=4, norm_type=None, res_connect=True):
        super().__init__()
        hidden_channels = int(in_ch * ratio)
        self.norm_type = norm_type
        self.res_connect = res_connect
        self.nonlinear = nnx.silu
        self.conv_inverted = nnx.Conv(in_ch, hidden_channels*2, kernel_size=1, strides=1, padding=0)
        self.conv_depth = nnx.Conv(hidden_channels * 2, hidden_channels*2, kernel_size=3, strides=1, padding=1, groups=hidden_channels*2)
        self.conv_point = nnx.Conv(hidden_channels, out_ch, kernel_size=1, strides=1, padding=0, use_bias=False)
        self.norm = None
        if norm_type == 'rms_norm':
            self.norm = nnx.RMSNorm(out_ch, epsilon=1e-5, elementwise_affine=True, bias=True, rngs=rngs)

    def __call__(self, x):
        if self.res_connect:
            residual = x
        
        x = self.conv_inverted(x)
        x = self.nonlinear(x)
        x = self.conv_depth(x)
        x, gate = jnp.split(x, 2, axis=1)
        x = x * self.nonlinear(gate)
        x = self.conv_point(x)

        if self.norm_type == 'rms_norm':
            x = self.norm(jnp.moveaxis(x, 1, -1))
            x = jnp.moveaxis(x, -1, 1)

        if self.res_connect:
            x = x + residual
        
        return x


class SanaModulatedNorm(nnx.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nnx.LayerNorm(dim, epsilon=eps)

    def __call__(self, x, temb, scale_shift_table):
        x = self.norm(x)
        shift, scale = jnp.split((scale_shift_table[None] + temb[:, None]), 2, axis=1)
        x = x * (1 + scale) + shift
        return x


class Attention(nnx.Module):
    def __init__(
        self,
        config: sana_config,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        qk_norm: Optional[str] = None,
        norm_num_groups: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        processor = None,
        out_dim: int = None,
        rngs=nnx.Rngs(0),
        elementwise_affine: bool = True,
    ):
        super().__init__()
        
        self.config = config
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.out_dim = out_dim if out_dim is not None else query_dim

        if norm_num_groups is not None:
            self.group_norm = nnx.GroupNorm(num_groups=norm_num_groups, epsilon=eps, feature_axes=1)
        else:
            self.group_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nnx.LayerNorm(self.inner_dim // self.heads, epsilon=eps, elementwise_affine=elementwise_affine, feature_axes=-1, rngs=rngs)
            self.norm_k = nnx.LayerNorm(self.inner_dim // self.heads, epsilon=eps, elementwise_affine=elementwise_affine, feature_axes=-1, rngs=rngs)
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim=self.inner_dim // self.heads, eps=eps, elementwise_affine=elementwise_affine, feature_axes=-1, rngs=rngs)
            self.norm_k = RMSNorm(dim=self.inner_dim // self.heads, eps=eps, elementwise_affine=elementwise_affine, feature_axes=-1, rngs=rngs)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None,'layer_norm','rms_norm'")

        self.to_q = nnx.Linear(query_dim, self.inner_dim, use_bias=bias, rngs=rngs)

        if not self.is_cross_attention:
            self.to_k = nnx.Linear(self.cross_attention_dim, self.inner_dim, use_bias=bias, rngs=rngs)
            self.to_v = nnx.Linear(self.cross_attention_dim, self.inner_dim, use_bias=bias, rngs=rngs)
        else:
            self.to_k = nnx.Linear(self.cross_attention_dim, self.inner_dim, use_bias=bias, rngs=rngs)
            self.to_v = nnx.Linear(self.cross_attention_dim, self.inner_dim, use_bias=bias, rngs=rngs)

        self.to_out = nnx.Linear(self.inner_dim, self.out_dim, use_bias=out_bias, rngs=rngs)
        
        self.processor = processor

    def __call__(
        self,
        hidden_states: Array,
        encoder_hidden_states: Optional[Array] = None,
        attention_mask: Optional[Array] = None,
    ) -> Array:
        
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )


from torch.nn import functional as F

class SanaLinearAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product linear attention.
    """

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        # Save the original data type to cast back at the end.
        original_dtype = hidden_states.dtype

        # If no encoder hidden states are provided, use the hidden_states for keys and values.
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Compute projections using the provided attention module.
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Assume query, key, value have shape [batch, seq_len, total_dim]
        # and that total_dim = attn.heads * head_dim.
        batch_size, seq_len, total_dim = query.shape
        heads = attn.heads
        head_dim = total_dim // heads

        # Reshape projections for multi-head attention.
        # For query: reshape from [B, S, total_dim] -> [B, heads, head_dim, S]
        query = jnp.transpose(query, (0, 2, 1))
        query = query.reshape(batch_size, heads, head_dim, seq_len)

        # For key: reshape from [B, S, total_dim] -> [B, heads, head_dim, S] then transpose to [B, heads, S, head_dim]
        key = jnp.transpose(key, (0, 2, 1))
        key = key.reshape(batch_size, heads, head_dim, seq_len)
        key = jnp.transpose(key, (0, 1, 3, 2))

        # For value: reshape from [B, S, total_dim] -> [B, heads, head_dim, S]
        value = jnp.transpose(value, (0, 2, 1))
        value = value.reshape(batch_size, heads, head_dim, seq_len)

        # Apply ReLU activation to query and key.
        query = jax.nn.relu(query)
        key = jax.nn.relu(key)

        # Cast to float32 for numerical stability.
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)
        value = value.astype(jnp.float32)

        # Pad the value tensor along the head_dim axis (axis=2) with one extra row of constant 1.0.
        # This changes its shape from [B, heads, head_dim, S] to [B, heads, head_dim+1, S].
        value = jnp.pad(
            value,
            ((0, 0), (0, 0), (0, 1), (0, 0)),
            mode="constant",
            constant_values=1.0,
        )

        # Compute the attention scores.
        # First, multiply value and key: shapes [B, heads, head_dim+1, S] x [B, heads, S, head_dim] -> [B, heads, head_dim+1, head_dim]
        scores = jnp.matmul(value, key)
        # Next, multiply the scores by the query: [B, heads, head_dim+1, head_dim] x [B, heads, head_dim, S] -> [B, heads, head_dim+1, S]
        hidden_states = jnp.matmul(scores, query)

        # The last element along axis=2 acts as a normalization term.
        # Divide the first head_dim elements by the normalization term (with a small constant for stability).
        numerator = hidden_states[:, :, :-1, :]
        denominator = hidden_states[:, :, -1:, :] + 1e-15
        hidden_states = numerator / denominator

        # Reshape back to [B, S, total_dim]:
        # First, combine the heads and head_dim dimensions.
        hidden_states = hidden_states.reshape(batch_size, heads * head_dim, seq_len)
        # Then transpose to [B, S, total_dim].
        hidden_states = jnp.transpose(hidden_states, (0, 2, 1))
        hidden_states = hidden_states.astype(original_dtype)

        # Apply final output projections.
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # If using float16, clip the output to the valid representable range.
        if original_dtype == jnp.float16:
            hidden_states = jnp.clip(hidden_states, -65504, 65504)

        return hidden_states


# jnp.pad()

class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True, feature_axes=-1, rngs=nnx.Rngs(0)):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.feature_axes = feature_axes
        if self.elementwise_affine:
            self.scale = nnx.Param(jnp.ones((dim,)))
            self.bias = nnx.Param(jnp.zeros((dim,)))

    def __call__(self, x):
        norm_x = jnp.sqrt(jnp.mean(jnp.square(x), axis=self.feature_axes, keepdims=True) + self.eps)
        out = x / norm_x
        if self.elementwise_affine:
            out = out * jnp.expand_dims(self.scale, axis=tuple(range(x.ndim -1)))
            out = out + jnp.expand_dims(self.bias, axis=tuple(range(x.ndim -1)))
        return out


class SanaTransformerBlock(nnx.Module):
    r"""
    Transformer block introduced in [Sana](https://huggingface.co/papers/2410.10629).
    """

    def __init__(
        self,
        config = sana_config,
        dim: int = 1152,
        num_attention_heads: int = 36,
        attention_head_dim: int = 32,
        dropout: float = 0.0,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        attention_bias: bool = True,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_out_bias: bool = True,
        mlp_ratio: float = 2.5,
        rngs=nnx.Rngs(0),
    ) -> None:
        super().__init__()

        # 1. Self Attention
        self.norm1 = nnx.LayerNorm(
            dim, elementwise_affine=False, epsilon=norm_eps, rngs=rngs
        )
        self.attn1 = Attention(
            sana_config, 
            query_dim=dim, cross_attention_dim=config.cross_attn_dim, 
            heads=config.attn_heads, dim_head=config.attn_head_dim,
            processor=SanaLinearAttnProcessor2_0
        )

        # 2. Cross Attention
        if cross_attention_dim is not None:
            self.norm2 = nnx.LayerNorm(
                dim,
                elementwise_affine=norm_elementwise_affine,
                epsilon=norm_eps,
                rngs=rngs,
            )
            self.attn2 = nnx.MultiHeadAttention(
                num_heads=num_cross_attention_heads,
                in_features=dim,
                features_per_head=cross_attention_head_dim,
                dropout_rate=dropout,
                use_bias=True,
                out_bias=attention_out_bias,
                cross_features=cross_attention_dim,
                rngs=rngs,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.ff = GlumbConv(
            dim, dim, mlp_ratio, norm_type=None, residual_connection=False, rngs=rngs
        )

        self.scale_shift_table = nnx.Param(
            jax.random.normal(rngs.params, (6, dim)) / (dim**0.5)
        )

    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Optional[Array] = None,
        encoder_hidden_states: Optional[Array] = None,
        encoder_attention_mask: Optional[Array] = None,
        timestep: Optional[Array] = None,
        height: int = None,
        width: int = None,
    ) -> Array:
        batch_size = hidden_states.shape[0]

        # 1. Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            self.scale_shift_table.value[None] + timestep.reshape(batch_size, 6, -1),
            6,
            axis=1,
        )

        # 2. Self Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        # norm_hidden_states = norm_hidden_states.to(hidden_states.dtype) #dtype already in jax

        attn_output = self.attn1(norm_hidden_states, mask=attention_mask)
        hidden_states = hidden_states + gate_msa * attn_output

        # 3. Cross Attention
        if self.attn2 is not None:
            attn_output = self.attn2(
                hidden_states, encoder_hidden_states, mask=encoder_attention_mask
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        norm_hidden_states = jnp.reshape(
            norm_hidden_states, (norm_hidden_states.shape[0], height, width, -1)
        )
        norm_hidden_states = jnp.moveaxis(norm_hidden_states, -1, 1)
        ff_output = self.ff(norm_hidden_states)
        ff_output = jnp.moveaxis(ff_output, 1, -1)
        ff_output = jnp.reshape(
            ff_output, (ff_output.shape[0], -1, ff_output.shape[-1])
        )

        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states
