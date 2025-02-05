import jax
from jax import Array, numpy as jnp, random as jrand
from flax import nnx

rngs = nnx.Rngs(1234)
xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0)


class Conv3(nnx.Module):
    def __init__(self, inch, outch, is_res=False):
        super().__init__()
        self.main_conv = nnx.Sequential(
            nnx.Conv(inch, outch, kernel_size=(3, 3), padding=1, kernel_init=xavier_init, bias_init=zero_init, rngs=rngs),
            nnx.GroupNorm(outch, 8, rngs=rngs),
            nnx.relu
        )
        self.conv = nnx.Sequential(
            nnx.Conv(outch, outch, (3, 3), padding=1, kernel_init=xavier_init, bias_init=zero_init, rngs=rngs),
            nnx.GroupNorm(outch, 8, rngs=rngs),
            nnx.relu,
            nnx.Conv(outch, outch, (3, 3), padding=1, kernel_init=xavier_init, bias_init=zero_init, rngs=rngs),
            nnx.GroupNorm(outch, 8, rngs=rngs),
            nnx.relu
        )
        self.is_res = is_res
        
    def __call__(self, x):
        x = self.main_conv(x)
        return (x + self.conv(x)) / 1.414 if self.is_res else self.conv(x)

class Interpolate(nnx.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def __call__(self, x: Array) -> Array:
        shape = (x.shape[0], x.shape[1] * self.scale_factor, x.shape[2] * self.scale_factor, x.shape[3])
        x = jax.image.resize(x, shape=shape, method="nearest")
        # print(f'interpolated {x.shape}')
        return x

from functools import partial

class UnetDown(nnx.Module):
    def __init__(self, inch, outch):
        super().__init__()
        self.model = nnx.Sequential(Conv3(inch, outch), partial(nnx.max_pool, window_shape=(2, 2)))
        
    def __call__(self, x):
        return self.model(x)

class UnetUp(nnx.Module):
    def __init__(self, inch, outch):
        super().__init__()
        self.model = nnx.Sequential(
            nnx.ConvTranspose(inch, outch, (2,2), strides=(2, 2), rngs=rngs),
            # Conv3(inch, outch),
            # Interpolate(),
            Conv3(outch, outch),
            Conv3(outch, outch)
        )
        
    def __call__(self, x, skip):
        out = jnp.concat((x, skip), axis=-1)
        return self.model(out)

class TimeSiren():
    def __init__(self, dim):
        super().__init__()
        self.lin_1 = nnx.Linear(1, dim, rngs=rngs, kernel_init=zero_init, use_bias=False)
        self.lin_2 = nnx.Linear(dim, dim, rngs=rngs, kernel_init=zero_init, bias_init=zero_init)

    def __call__(self, x: Array):
        x = x.reshape(-1, 1)
        x = jnp.sin(self.lin_1(x))
        x = self.lin_2(x)

        return x

class MiniUnet(nnx.Module):
    def __init__(self, dim=128, in_channels=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dim = dim

        self.init_conv = Conv3(in_channels, dim, is_res=True)

        self.down_1 = UnetDown(dim, dim)
        self.down_2 = UnetDown(dim, 2 * dim)
        self.down_3 = UnetDown(2*dim, 2*dim)

        avgpool = partial(nnx.avg_pool, window_shape=(4, 4))
        self.to_vec = nnx.Sequential(avgpool, nnx.relu)

        self.time_embed = TimeSiren(2*dim)

        self.up_0 = nnx.Sequential(
            nnx.ConvTranspose(2*dim, 2*dim, (4,4), strides=(4, 4), rngs=rngs),
            nnx.GroupNorm(2*dim, 8, rngs=rngs),
            nnx.relu
        )

        self.up_1 = UnetUp(4 * dim, 2 * dim)
        self.up2 = UnetUp(4 * dim, dim)
        self.up3 = UnetUp(2 * dim, dim)

        self.out = nnx.Conv(2*dim, self.out_channels, (3, 3), padding=1, rngs=rngs)

    def __call__(self, x, t, y):
        x = self.init_conv(x)

        down1 = self.down_1(x)
        down2 = self.down_2(down1)
        down3 = self.down_3(down2)

        thro = self.to_vec(down3)
        temb = self.time_embed(t).reshape(-1, 1, 1, self.n_feat * 2)

        thro = self.up_0(thro + temb + y)

        up1 = self.up_1(thro, down3) + temb
        up2 = self.up_2(up1, down2)
        up3 = self.up_3(up2, down1)

        out = self.out(jnp.concat((up3, x), axis=-1))

        return out
