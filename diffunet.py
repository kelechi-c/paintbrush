"""
Diffusion U-Net in JAX. adapted from https://github.com/lucidrains/rectified-flow-pytorch
"""

import jax, flax, einops
from jax import Array, numpy as jnp, random as jrand
from flax import nnx
from einops import rearrange, repeat
from einops.layers.flax import Rearrange
from functools import partial

rngs = nnx.Rngs(333)
randkey = jrand.key(333)

class RMSNorm(nnx.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nnx.Param(jnp.zeros(dim))
        
    def __call__(self, x: Array) -> Array:
        return x * (self.gamma.value + 1) * self.scale

class SinusoidPosEmbedding(nnx.Module):
    def __init__(self, dim, theta):
        super().__init__()
        self.dim = dim 
        self.theta = theta

    def __call__(self, x: Array) -> Array:
        half_dim = self.dim // 2
        emb = jnp.log(self.theta) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x * emb
        emb = jnp.concatenate([emb.sin(), emb.cos()], axis=-1)
        return emb


class LabelEmbedder(nnx.Module):
    def __init__(self, num_classes, hidden_size, drop):
        super().__init__()
        use_cfg_embeddings = drop > 0
        self.embedding_table = nnx.Embed(
            num_classes + int(use_cfg_embeddings),
            hidden_size,
            rngs=rngs,
            embedding_init=nnx.initializers.normal(0.02),
        )
        self.num_classes = num_classes
        self.dropout = drop

    def token_drop(self, labels, force_drop_ids=None) -> Array:
        if force_drop_ids is None:
            drop_ids = jrand.normal(key=randkey, shape=labels.shape[0])
        else:
            drop_ids = force_drop_ids == 1

        labels = jnp.where(drop_ids, self.num_classes, labels)

        return labels

    def __call__(self, labels, train: bool = True, force_drop_ids=None) -> Array:
        use_drop = self.dropout > 0
        if (train and use_drop) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        label_embeds = self.embedding_table(labels)

        return label_embeds

class Interpolate(nnx.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
    
    def __call__(self, x: Array) -> Array:
        shape = (x.shape[0], x.shape[2] * self.scale_factor, x.shape[3] * self.scale_factor, x.shape[1])
        x = jax.image.resize(x, shape=shape, method="nearest")
        return x

def upsample(dim, dim_out = None):
    return nnx.Sequential(
        Interpolate(),
        nnx.Conv(dim, dim_out or dim, kernel_size=(3, 3), rngs=rngs),
    )

def downsample(dim, dim_out = None):
    return nnx.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nnx.Conv(dim * 4, dim_out or dim, kernel_size=(1, 1), rngs=rngs)
    )

class ConvBlock(nnx.Module):
    def __init__(self, dim, dim_out, drop=0.0):
        super().__init__()
        self.project = nnx.Conv(dim, dim_out, kernel_size=(3, 3), rngs=rngs)
        self.norm = RMSNorm(dim_out)
        self.droplayer = nnx.Dropout(drop, rngs=rngs)
        
    def __call__(self, x: Array, scale_shift=None) -> Array:
        x = self.norm(self.project(x))
        
        if scale_shift:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = nnx.silu(x)
        x = self.droplayer(x)
        return x

class ResBlock(nnx.Module):
    def __init__(self, dim, dim_out, *, time_dim=None, drop=0.0):
        super().__init__()
        self.mlp = nnx.Linear(time_dim, dim_out, rngs=rngs) if time_dim else None
        self.block_1 = ConvBlock(dim, dim_out, drop=drop)
        self.block_2 = ConvBlock(dim_out, dim_out, drop=drop)
        
    def __call__(self, x: Array, time_emb: Array = None) -> Array:
        scale_shift = None
        
        if self.mlp and time_emb :
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = jnp.array_split(time_emb, 2, axis=1)
        
        h = self.block_1(x, scale_shift = scale_shift)
        
        h = self.block_2(h)
        
        return h + x


class LinearAttention(nnx.Module):
    def __init__(self, dim, heads=4, head_dim=32, num_mem_kv=4):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.heads = heads
        hidden_dim = head_dim * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nnx.Param(jnp.zeros((2, heads, num_mem_kv, head_dim)))
        self.qkv_project = nnx.Conv(dim, hidden_dim * 3, kernel_size=(1, 1), rngs=rngs)
        self.out_project = nnx.Sequential(
            nnx.Conv(hidden_dim, dim, 1, rngs=rngs),
            RMSNorm(dim)
        )

    def __call__(self, x: Array):
        b, h, w, c = x.shape
        x = self.norm(x)

        qkv = jnp.array_split(self.qkv_project(x), 3, axis=1)
        q, k, v = map(
            lambda t: rearrange(t, "b x y (h c) -> b h (x y) c", h=self.heads), qkv
        )

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv.value)
        k, v = map(partial(jnp.concatenate, axis = -2), ((mk, k), (mv, v)))

        q = q * self.scale
        context = jnp.einsum('bhdn, bhen -> bhde', k, v)
        out = jnp.einsum('bhde, bhdn -> bhen', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)

        return self.out_project(out)


class Attention(nnx.Module):
    def __init__(self, dim, heads=4, head_dim=32, num_mem_kv=4, flash=False):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.heads = heads 
        hidden_dim = head_dim * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nnx.Param(jnp.zeros((2, heads, num_mem_kv, head_dim)))
        self.qkv_project = nnx.Conv(dim, hidden_dim * 3, kernel_size=(1, 1), rngs=rngs)
        self.out_project = nnx.Conv(hidden_dim, dim, (1, 1), rngs=rngs)

    def __call__(self, x: Array):
        b, h, w, c = x.shape
        x = self.norm(x)

        qkv = jnp.array_split(self.qkv_project(x), 3, axis=1)

        q, k, v = map(
            lambda t: rearrange(t, "b x y (h c) -> b h (x y) c", h=self.heads), qkv
        )

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv.value)
        k, v = map(partial(jnp.concatenate, axis = -2), ((mk, k), (mv, v)))

        q = q * self.scale

        sim = jnp.einsum('bhid, bhjd -> bhij', q, k)
        attn_logs = nnx.softmax(sim, axis = -1)
        out = jnp.einsum('bhij, bhjd -> bhid', attn_logs, q)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = self.heads, x = h, y = w)

        return self.out_project(out)

class TimeMLP(nnx.Module):
    def __init__(self, time_dim, fourier_dim, sinusoid_emb):
        super().__init__()
        self.lin_1 = nnx.Linear(fourier_dim, time_dim, rngs=rngs)
        self.sinusoid_embed = sinusoid_emb
        self.lin_2 = nnx.Linear(time_dim, time_dim)

    def __call__(self, x):
        x = self.lin_1(self.sinusoid_embed(x))
        x = self.lin_2(nnx.gelu(x))

        return x


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)

# denoising U-Net model
class Unet(nnx.Module):
    def __init__(
        self, 
        dim, init_dim=None, out_dim=None, 
        dim_mults=(1, 2, 4, 8), channels=4,
        learn_var=False,
        sinusoid_theta=10_000,
        drop=0.0, attn_head_dim=32, attn_heads=4,
        full_attn=None, num_res_stream=2
    ):
        super().__init__()

        self.channels = channels
        init_dim = init_dim or dim
        self.init_conv = nnx.Conv(
            channels, init_dim,
            kernel_size=(7, 7),
            padding=3
        )

        dims = [init_dim, *map(lambda m: dim*m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'{dims = } / {in_out = }')

        time_dim = dim * 4

        sinusoid_posembed = SinusoidPosEmbedding(dim, theta=sinusoid_theta)
        fourier_dim = dim

        self.time_mlp = TimeMLP(time_dim, fourier_dim, sinusoid_posembed)
        self.label_embed = LabelEmbedder(102, dim)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_head_dim = cast_tuple(attn_head_dim, num_stages)

        block_args = (in_out, full_attn, attn_heads, attn_head_dim)

        assert len(full_attn) == len(dim_mults), 'full_attn must match num of mulipliers'

        # FullAttn = partial(Attention, flash=False)
        resnet_block = partial(ResBlock, time_dim=time_dim, drop=drop)

        res_conv = partial(nnx.Conv, kernel_size=(1, 1), use_bias=False)

        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        for idx, ((dim_in, dim_out), layer_attn, layer_heads, layer_head_dim) in enumerate(zip(block_args)):
            is_last = idx >= (num_resolutions-1)

            attn_block = Attention if layer_attn else LinearAttention

            self.downs.append([
                resnet_block(dim_in, dim_in), resnet_block(dim_in, dim_in), 
                attn_block(dim_in, layer_heads, layer_head_dim),
                downsample(dim_in, dim_out) if not is_last else nnx.Conv(dim_in, dim_out, (3,3), padding=1)
            ])

        mid_dim = dims[-1]
        self.mid_block_1 = ResBlock(mid_dim, mid_dim, time_dim=time_dim)
        self.mid_attention = Attention(mid_dim, attn_heads[-1], attn_head_dim[-1])
        self.mid_block_2 = ResBlock(mid_dim, mid_dim)

        for idx, ((dim_in, dim_out), layer_attn, layer_heads, layer_head_dim) in enumerate(zip(*map(reversed, block_args))):
            is_last = idx == (len(in_out) - 1)

            attn_block = Attention if layer_attn else LinearAttention

            self.ups.append([
                ResBlock(dim_out+dim_in, dim_out),
                ResBlock(dim_out+dim_in, dim_out),
                attn_block(dim_out, layer_heads, layer_head_dim),
                upsample(dim_out, dim_in)
            ])

        default_out_dim = channels * (1 if not learn_var else 2)
        self.out_dim = out_dim or default_out_dim

        self.finres_transform = res_conv(init_dim*2, init_dim)
        self.final_resblock = ResBlock(init_dim*2, init_dim)
        self.final_conv = nnx.Conv(init_dim, self.out_dim, kernel_size=(1, 1))

    def __call__(self, x: Array, t: Array, y: Array):
        x = self.init_conv(x)
        res = x
        h_skips = []

        y = self.label_embed(y)
        t = self.time_mlp(t) + y

        for block1, block2, attn, downsampler in self.downs:
            x = block1(x, t)
            h_skips.append(x)

            x = attn(block2(x, t))
            h_skips.append(x)
            x = downsampler(x)

        x = self.mid_block_1(x, t)
        x = self.mid_attention(x)
        x = self.mid_block_2(x, t)

        for block1, block2, attn, upsample_block in self.ups:
            x = jnp.concat([x, h_skips.pop()], axis=1)
            x = block1(x, t)
            x = jnp.concat([x, h_skips.pop()], axis=1)
            x = attn(block2(x, t))
            x = upsample_block(x)

        x = jnp.concat([x, res], axis=1)
        x = self.final_resblock(x, t)
        x = self.final_conv(x)

        return x


# wrapper for flow matching loss and sampling
class FlowWrapper(nnx.Module):
    def __init__(self, model: Unet):
        self.model = model

    def __call__(self, x: Array, c: Array) -> Array:
        img_latents, cond = x, c

        x_1, c = img_latents, cond  # reassign to more concise variables
        bs = x_1.shape[0]

        x_0 = jrand.normal(randkey, x_1.shape)  # noise
        t = jrand.uniform(randkey, (bs,))
        t = nnx.sigmoid(t)

        inshape = [1] * len(x_1.shape[1:])
        t_exp = t.reshape([bs, *(inshape)])

        x_t = (1 - t_exp) * x_0 + t_exp * x_1
        dx_t = x_1 - x_0  # actual vector/velocity difference

        vtheta = self.model(
            x_t, t, c, mask_ratio=self.mask_ratio
        )  # model vector prediction

        mean_dim = list(
            range(1, len(x_1.shape))
        )  # across all dimensions except the batch dim
        mean_square = (dx_t - vtheta) ** 2  # squared difference/error
        batchwise_mse_loss = jnp.mean(mean_square, axis=mean_dim)  # mean loss
        loss = jnp.mean(batchwise_mse_loss)

        return loss.mean()

    def flow_step(self, x_t: Array, cond: Array, t_start: float, t_end: float) -> Array:
        """Performs a single flow step using Euler's method."""
        t_mid = (t_start + t_end) / 2.0
        # Broadcast t_mid to match x_t's batch dimension
        t_mid = jnp.full((x_t.shape[0],), t_mid)
        # Evaluate the vector field at the midpoint
        v_mid, _ = self.model(x=x_t, y=cond, t=t_mid)
        # Update x_t using Euler's method
        x_t_next = x_t + (t_end - t_start) * v_mid

        return x_t_next

    def sample(self, label: Array, num_steps: int = 50):
        """Generates samples using flow matching."""
        time_steps = jnp.linspace(0.0, 1.0, num_steps + 1)
        x_t = jax.random.normal(randkey, (len(label), 32, 32, 4))  # important change

        for k in tqdm(range(num_steps), desc="Sampling images"):
            x_t = self.flow_step(
                x_t=x_t, cond=label, t_start=time_steps[k], t_end=time_steps[k + 1]
            )
            

        return x_t / 0.13025
