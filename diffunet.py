"""
Diffusion U-Net in JAX. adapted from https://github.com/cloneofsimo/minDiffusion and https://github.com/lucidrains/rectified-flow-pytorch
"""

import jax, flax, einops
from jax import Array, numpy as jnp, random as jrand
from flax import nnx
from einops import rearrange
from einops.layers.flax import Rearrange


rngs = nnx.Rngs(333)

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
        
        return h
    

