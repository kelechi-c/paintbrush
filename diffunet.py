"""
Diffusion U-Net in JAX. adapted from https://github.com/lucidrains/rectified-flow-pytorch
"""

import jax
from jax import Array, numpy as jnp, random as jrand
from flax import nnx
from einops import rearrange, repeat
from einops.layers.flax import Rearrange
from functools import partial
from tqdm.auto import tqdm

rngs = nnx.Rngs(333)
randkey = jrand.key(333)

rngs = nnx.Rngs(333)
randkey = jrand.key(333)


class RMSNorm(nnx.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nnx.Param(jnp.zeros(dim))

    def __call__(self, x: Array) -> Array:
        return x * (self.gamma.value + 1) * self.scale


class SinusoidPosEmbedding(nnx.Module):
    def __init__(self, dim, theta):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, x: Array) -> Array:
        # print(f'sinusoid in {x.shape}')
        half_dim = self.dim // 2
        emb = jnp.log(self.theta) / (half_dim - 1)

        # Use jnp.arange for JAX compatibility
        emb = jnp.exp(jnp.arange(half_dim) * -emb)

        # Use jnp.einsum for element-wise multiplication and broadcasting
        emb = jnp.einsum("i,j->ij", x, emb)

        # Use jnp.concatenate for concatenation along the last dimension
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        # print(f'sinus embed {emb.shape = }')
        return emb


class LabelEmbedder(nnx.Module):
    def __init__(self, num_classes, hidden_size, drop=0.0):
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
        shape = (
            x.shape[0],
            x.shape[1] * self.scale_factor,
            x.shape[2] * self.scale_factor,
            x.shape[3],
        )
        x = jax.image.resize(x, shape=shape, method="nearest")
        # print(f'interpolated {x.shape}')
        return x


def upsample(dim, dim_out=None):
    return nnx.Sequential(
        Interpolate(),
        nnx.Conv(dim, dim_out or dim, kernel_size=(3, 3), rngs=rngs),
    )


# def downsample(dim, dim_out = None):
#     return nnx.Sequential(
#         Rearrange('b (h 2) (w 2) c -> b h w (c 2 2)'),
#         nnx.Conv(dim * 4, dim_out or dim, kernel_size=(1, 1), rngs=rngs)
#     )
class downsample(nnx.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv = nnx.Conv(dim * 4, dim_out or dim, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = rearrange(x, "b (h p1) (w p2) c -> b h w (c p1 p2)", p1=2, p2=2)
        x = self.conv(x)
        # print(f'downsampled {x.shape}')
        return x


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
            # print(f'{scale.shape = } / {x.shape = }')
            x = x * (scale + 1) + shift

        x = nnx.silu(x)
        x = self.droplayer(x)
        return x


class ResBlock(nnx.Module):
    def __init__(self, dim, dim_out, *, time_dim=None, drop=0.0, is_res=True):
        super().__init__()
        self.mlp = nnx.Linear(time_dim, dim_out * 2, rngs=rngs) if time_dim else None
        self.block_1 = ConvBlock(dim, dim_out, drop=drop)
        self.block_2 = ConvBlock(dim_out, dim_out, drop=drop)
        self.isres = is_res

    def __call__(self, x: Array, time_emb: Array = None) -> Array:
        scale_shift = None
        res = x
        if self.mlp and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b 1 1 c")
            scale_shift = jnp.array_split(time_emb, 2, axis=-1)

        h = self.block_1(x, scale_shift=scale_shift)

        h = self.block_2(h)
        # print(f'resblock {h.shape = } / {x.shape = }')
        # print(f'{h.shape = } / {res.shape = }')
        return h + res if self.isres else h


class LinearAttention(nnx.Module):
    def __init__(self, dim, heads=4, head_dim=32, num_mem_kv=4):
        super().__init__()
        self.scale = head_dim**-0.5
        self.heads = heads
        hidden_dim = head_dim * heads
        # print(f'{heads = } / {head_dim = } / {hidden_dim = }')

        self.norm = RMSNorm(dim)

        self.mem_kv = nnx.Param(jnp.zeros((2, heads, num_mem_kv, head_dim)))
        self.qkv_project = nnx.Conv(dim, hidden_dim * 3, kernel_size=(1, 1), rngs=rngs)
        self.out_project = nnx.Sequential(
            nnx.Conv(hidden_dim, dim, 1, rngs=rngs), RMSNorm(dim)
        )

    def __call__(self, x: Array):
        b, h, w, c = x.shape
        print(f"linear attn in {x.shape = }")

        x = self.norm(x)

        qkv = self.qkv_project(x)
        print(f"{qkv.shape = }")
        qkv = jnp.array_split(qkv, 3, axis=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b x y (h c) -> b h c (x y)", h=self.heads), qkv
        )
        print(f"{q.shape = }")

        # mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv.value)
        # k, v = map(partial(jnp.concatenate, axis = -2), ((mk, k), (mv, v)))
        # print(f'linear {k.shape = } / {v.shape = }')

        q = q * self.scale
        # context = jnp.einsum('bhdn, bhen -> bhde', k, v)
        # out = jnp.einsum('bhde, bhdn -> bhen', context, q)

        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        print(f"linear {out.shape = }")

        return self.out_project(out)


class Attention(nnx.Module):
    def __init__(self, dim, heads=4, head_dim=32, num_mem_kv=4, flash=False):
        super().__init__()
        self.scale = head_dim**-0.5
        self.heads = heads
        hidden_dim = head_dim * heads
        # print(f'{heads = } / {head_dim = } / {hidden_dim = }')

        self.norm = RMSNorm(dim)

        self.mem_kv = nnx.Param(jnp.zeros((2, heads, num_mem_kv, head_dim)))
        self.qkv_project = nnx.Conv(dim, hidden_dim * 3, kernel_size=(1, 1), rngs=rngs)
        self.out_project = nnx.Conv(hidden_dim, dim, (1, 1), rngs=rngs)

    def __call__(self, x: Array):
        b, h, w, c = x.shape
        # print(f'attn in - {x.shape = }')

        x = self.norm(x)

        qkv = jnp.array_split(self.qkv_project(x), 3, axis=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b x y (h c) -> b h (x y) c", h=self.heads), qkv
        )

        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv.value)
        k, v = map(partial(jnp.concat, axis=-2), ((mk, k), (mv, v)))
        # print(f'attn {k.shape = } / {v.shape = }')

        q = q * self.scale

        sim = jnp.einsum("bhid, bhjd -> bhij", q, k)
        attn_logs = nnx.softmax(sim, axis=-1)
        out = jnp.einsum("bhij, bhjd -> bhid", attn_logs, v)

        out = rearrange(out, "b h (x y) d -> b x y (h d)", h=self.heads, x=h, y=w)

        # print(f'attn pre {out.shape = }')

        return self.out_project(out)


class TimeMLP(nnx.Module):
    def __init__(self, time_dim, fourier_dim, sinusoid_emb):
        super().__init__()
        self.lin_1 = nnx.Linear(fourier_dim, time_dim, rngs=rngs)
        self.sinusoid_embed = sinusoid_emb
        self.lin_2 = nnx.Linear(time_dim, time_dim, rngs=rngs)

    def __call__(self, x):
        x = self.lin_1(self.sinusoid_embed(x))
        x = self.lin_2(nnx.gelu(x))
        # print(f'time embed - {x.shape = }')

        return x


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


# denoising U-Net model
class Unet(nnx.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=4,
        learn_var=False,
        sinusoid_theta=10_000,
        drop=0.0,
        attn_head_dim=32,
        attn_heads=4,
        full_attn=None,
        num_res_stream=2,
    ):
        super().__init__()

        self.channels = channels
        init_dim = init_dim or dim
        self.init_conv = nnx.Conv(
            channels, init_dim, kernel_size=(7, 7), padding=3, rngs=rngs
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"{dims = } / {in_out = }")

        time_dim = dim

        sinusoid_posembed = SinusoidPosEmbedding(dim, theta=sinusoid_theta)
        fourier_dim = dim

        self.time_mlp = TimeMLP(time_dim, fourier_dim, sinusoid_posembed)
        self.label_embed = LabelEmbedder(102, dim)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_head_dim = cast_tuple(attn_head_dim, num_stages)

        block_args = (in_out, full_attn, attn_heads, attn_head_dim)

        assert len(full_attn) == len(
            dim_mults
        ), "full_attn must match num of mulipliers"

        # FullAttn = partial(Attention, flash=False)
        resnet_block = partial(ResBlock, time_dim=time_dim, drop=drop)

        res_conv = partial(nnx.Conv, kernel_size=(1, 1), use_bias=False, rngs=rngs)
        nnx_attn = partial(
            nnx.MultiHeadAttention, rngs=rngs, decode=False, dtype=jnp.bfloat16
        )

        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        for idx, (
            (dim_in, dim_out),
            layer_attn,
            layer_heads,
            layer_head_dim,
        ) in enumerate(zip(*block_args)):
            is_last = idx >= (num_resolutions - 1)

            attn_block = Attention if layer_attn else nnx_attn

            self.downs.append(
                [
                    resnet_block(dim_in, dim_in),
                    resnet_block(dim_in, dim_in),
                    (
                        attn_block(dim_in, layer_heads, layer_head_dim)
                        if layer_attn
                        else attn_block(layer_heads, dim_in, dim_in)
                    ),
                    (
                        downsample(dim_in, dim_out)
                        if not is_last
                        else nnx.Conv(dim_in, dim_out, (3, 3), padding=1, rngs=rngs)
                    ),
                ]
            )

        mid_dim = dims[-1]
        self.mid_block_1 = ResBlock(mid_dim, mid_dim)
        self.mid_attention = Attention(mid_dim, attn_heads[-1], attn_head_dim[-1])
        self.mid_block_2 = ResBlock(mid_dim, mid_dim)

        for idx, (
            (dim_in, dim_out),
            layer_attn,
            layer_heads,
            layer_head_dim,
        ) in enumerate(zip(*map(reversed, block_args))):
            is_last = idx == (len(in_out) - 1)
            print(map(reversed, block_args))

            attn_block = Attention if layer_attn else nnx_attn

            self.ups.append(
                [
                    resnet_block(dim_out + dim_in, dim_out, is_res=False),
                    resnet_block(dim_out + dim_in, dim_out, is_res=False),
                    (
                        attn_block(dim_out, layer_heads, layer_head_dim)
                        if layer_attn
                        else attn_block(layer_heads, dim_out, dim_out)
                    ),
                    (
                        upsample(dim_out, dim_in)
                        if not is_last
                        else nnx.Conv(dim_in, dim_out, (3, 3), padding=1, rngs=rngs)
                    ),
                ]
            )

        default_out_dim = channels * (1 if not learn_var else 2)
        self.out_dim = out_dim or default_out_dim

        self.finres_transform = res_conv(init_dim * 2, init_dim)
        self.final_resblock = resnet_block(init_dim, init_dim)
        self.final_conv = nnx.Conv(
            init_dim, self.out_dim, kernel_size=(1, 1), rngs=rngs
        )

    def __call__(self, x: Array, t: Array, y: Array):
        x = self.init_conv(x)
        res = x
        # print(f'{res.shape = }')
        h_skips = []

        y = self.label_embed(y)
        t = self.time_mlp(t) + y
        # print(f'{t.shape = } / {y.shape = }')
        for block1, block2, attn, downsample_block in self.downs:
            x = block1(x, t)
            # print(f'x after block1 {x.shape}')

            h_skips.append(x)

            x = attn(block2(x, t))
            # print(f'x after attn/block 2 {x.shape}')
            h_skips.append(x)
            x = downsample_block(x)
            # print(f'downsample {x.shape}')

        # print(f'downblocks {x.shape}')

        x = self.mid_block_1(x, t)
        x = self.mid_attention(x)
        x = self.mid_block_2(x, t)

        # print(f'midblocks {x.shape}')

        for block1, block2, attn, upsample_block in self.ups:
            # print(f'x preconcat {x.shape}')
            x = jnp.concat([x, h_skips.pop()], axis=-1)
            # print(f'x postconcat {x.shape}')

            x = block1(x, t)
            # print(f'x block1 {x.shape}')

            x = jnp.concat([x, h_skips.pop()], axis=-1)
            x = attn(block2(x, t))

            # print(f'x block2 {x.shape}')

            x = upsample_block(x)
            # print(f'upsample {x.shape}')

        x = jnp.concat([x, res], axis=-1)
        x = self.finres_transform(x)
        # print(f'tranformed {x.shape}')
        x = self.final_resblock(x, t)
        # print(f'finres = {x.shape}')
        x = self.final_conv(x)
        # print(f'final {x.shape}')
        return x


# wrapper for flow matching loss and sampling
class FlowMatch(nnx.Module):
    def __init__(self, model: Unet):
        super().__init__()
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

        vtheta = self.model(x_t, t, c)  # model vector prediction

        mean_dim = list(
            range(1, len(x_1.shape))
        )  # across all dimensions except the batch dim
        mean_square = (dx_t - vtheta) ** 2  # squared difference/error
        batchwise_mse_loss = jnp.mean(mean_square, axis=mean_dim)  # mean loss
        loss = jnp.mean(batchwise_mse_loss)

        return loss.mean()

    def flow_step(
        self, x_t: Array, cond: Array, t_start: float, t_end: float, cfg_scale=2.0
    ) -> Array:
        """Performs a single flow step using Euler's method."""
        t_mid = (t_start + t_end) / 2.0
        # Broadcast t_mid to match x_t's batch dimension
        t_mid = jnp.full((x_t.shape[0],), t_mid)
        # Evaluate the vector field at the midpoint
        v_mid = self.model(x=x_t, y=cond, t=t_mid)

        # classifier-free guidance sampling
        null_cond = jnp.zeros_like(cond)
        v_uncond = self.model(x_t, null_cond, t_mid)
        v_mid = v_uncond + cfg_scale * (v_mid - v_uncond)

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
