import jax, optax, wandb, torch, os, click, math, gc, time
import numpy as np
from flax import nnx
from jax import Array, numpy as jnp, random as jrand
from jax.sharding import NamedSharding, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from tqdm import tqdm
from einops import rearrange
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
import flax.traverse_util, pickle
from flax.serialization import to_state_dict, from_state_dict
from flax.core import freeze, unfreeze

import warnings
warnings.filterwarnings("ignore")
jax.config.update("jax_default_matmul_precision", "bfloat16")

JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20


class config:
    seed = 333
    batch_size = 128
    data_split = 10000
    img_size = 32
    patch_size = (2, 2)
    lr = 2e-4
    cfg_scale = 2.0
    vaescale_factor = 0.13025


num_devices = jax.device_count()
devices = jax.devices()

print(f"found {num_devices} JAX device(s)")
for device in devices:
    print(f"{device} / ")

mesh_devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(mesh_devices, axis_names="axis")
sharding = NamedSharding(mesh, PS("axis"))

# sd VAE for decoding latents
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.bfloat16
)
flower_data = load_dataset("tensorkelechi/latent_flowers102", split="train")
print("loaded vae")


def jax_collate(batch):
    latents = jnp.stack([jnp.array(item["latent"]) for item in batch], axis=0)
    labels = jnp.stack([int(item["label"]) for item in batch], axis=0)

    return {
        "latent": latents,
        "label": labels,
    }


@click.command()
@click.option("-r", "--run", default="single_batch")
@click.option("-e", "--epochs", default=config.epochs)
@click.option("-bs", "--batch_size", default=config.batch_size)
def main(run, epochs, batch_size):

    model = None
    rf_engine = None

    n_params = sum([p.size for p in jax.tree.leaves(nnx.state(model))])
    print(f"model parameters count: {n_params/1e6:.2f}M, ")

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=config.lr))

    train_loader = DataLoader(
        flower_data,
        batch_size=batch_size,
        num_workers=num_devices,
        drop_last=True,
        collate_fn=jax_collate,
    )

    sp = next(iter(train_loader))
    print(f"loaded data \n data sample: {sp['vae_output'].shape}")

    if run == "single_batch":
        model, loss = batch_trainer(
            epochs, model=model, optimizer=optimizer, train_loader=train_loader
        )
        wandb.finish()
        print(f"single batch training ended at loss: {loss:.4f}")

    elif run == "train":
        print(f"you missed your train looop impl boy")


main()
