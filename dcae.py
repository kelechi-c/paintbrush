import jax, math
from jax import Array, numpy as jnp, random as jrand
from flax import nnx


class ResBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch_norm",
    ) -> None:
        super().__init__()

        self.norm_type = norm_type

        self.nonlinearity = nnx.relu6
        self.conv1 = nnx.Conv(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nnx.Conv(in_channels, out_channels, 3, 1, 1, use_bias=False)
        self.norm = nnx.BatchNorm(out_channels)
        # self.norm = get_normalization(norm_type, out_channels)

    def __call__(self, hidden_states: Array) -> Array:
        residual = hidden_states
        
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.norm(hidden_states)

        return hidden_states + residual


class 