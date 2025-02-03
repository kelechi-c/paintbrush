import jax, optax, wandb, torch, os, click, math, gc, time
import numpy as np
from flax import nnx
from jax import Array, numpy as jnp, random as jrand
from jax.sharding import NamedSharding as NS, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from tqdm import tqdm
from functools import partial
from einops import rearrange
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt

import flax.traverse_util, pickle
from flax.serialization import to_state_dict, from_state_dict
from flax.core import freeze, unfreeze

from diffunet import Unet, FlowMatch

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
mesh = Mesh(mesh_devices, axis_names=("data"))
data_sharding = NS(mesh, PS("data"))
rep_sharding = NS(mesh, PS())


# sd VAE for decoding latents
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.bfloat16
)
flower_data = load_dataset("tensorkelechi/latent_flowers102", split="train")
print("loaded vae")


def jax_collate(batch):
    latents = jnp.stack(
        [jnp.array(item["latents"], dtype=jnp.bfloat16) for item in batch], axis=0
    )
    labels = jnp.stack(
        [jnp.array(int(item["label"]), dtype=jnp.int32) for item in batch], axis=0
    )

    return {
        "latent": latents,
        "label": labels,
    }


def wandb_logger(key: str, project_name, run_name=None):
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(
        project=project_name,
        name=run_name or None,
        settings=wandb.Settings(init_timeout=120),
    )


def device_get_model(model):
    state = nnx.state(model)
    state = jax.device_get(state)
    nnx.update(model, state)

    return model


def vae_decode(latent, vae=vae):
    tensor_img = rearrange(latent, "b h w c -> b c h w")
    tensor_img = torch.from_numpy(np.array(tensor_img))
    x = vae.decode(tensor_img).sample

    img = VaeImageProcessor().postprocess(
        image=x.detach(), do_denormalize=[True, True]
    )[0]

    return img


def process_img(img):
    img = vae_decode(img[None])
    return img


def image_grid(pil_images, file, grid_size=(3, 3), figsize=(10, 10)):
    rows, cols = grid_size
    assert len(pil_images) <= rows * cols, "Grid size must accommodate all images."

    # Create a matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten for easy indexing

    for i, ax in enumerate(axes):
        if i < len(pil_images):
            # Convert PIL image to NumPy array and plot
            ax.imshow(np.array(pil_images[i]))
            ax.axis("off")  # Turn off axis labels
        else:
            ax.axis("off")  # Hide empty subplots for unused grid spaces

    plt.tight_layout()
    plt.savefig(file, bbox_inches="tight")
    plt.show()

    return file


def sample_image_batch(step, model, labels):
    pred_model = device_get_model(model)
    pred_model.eval()

    image_batch = pred_model.sample(labels)
    file = f"fmsamples/{step}_flowdit.png"
    batch = [process_img(x) for x in image_batch]

    gridfile = image_grid(batch, file)
    print(f"sample saved @ {gridfile}")

    model.train()

    return gridfile


@partial(
    nnx.jit,
    in_shardings=(rep_sharding, rep_sharding, data_sharding),
    out_shardings=(None, None),
)
def train_step(model, optimizer, batch):

    def loss_func(model, batch):
        img_latents, label = batch["latent"], batch["label"]
        img_latents = rearrange(img_latents, "b c h w -> b h w c") * 0.13025
        loss = model(img_latents, label)

        return loss

    gradfn = nnx.value_and_grad(loss_func)
    loss, grads = gradfn(model, batch)
    grad_norm = optax.global_norm(grads)

    optimizer.update(grads)

    return loss, grad_norm


def batch_trainer(epochs, model, optimizer, train_loader, schedule):
    train_loss = 0.0
    model.train()

    batch = next(iter(train_loader))

    # wandb_logger(key="yourkey", project_name="mini_diffusion")

    stime = time.time()

    print("start overfitting.../")
    for epoch in tqdm(range(epochs)):
        lr = schedule(epoch)

        train_loss, grad_norm = train_step(model, optimizer, batch)
        print(
            f"step {epoch}, loss-> {train_loss.item():.4f}, grad_norm {grad_norm.item()}"
        )

        wandb.log(
            {
                "loss": train_loss.item(),
                "log_loss": math.log10(train_loss.item() + 1e-8),
                "grad_norm": grad_norm.item(),
                "log_grad_norm": math.log10(grad_norm.item() + 1e-8),
                "lr": lr,
            }
        )

        if epoch % 25 == 0:
            gridfile, sample_error = sample_image_batch(epoch, model, batch)
            image_log = wandb.Image(gridfile)
            wandb.log({"image_sample": image_log, "sample_error": sample_error})

        jax.clear_caches()
        gc.collect()

    etime = time.time() - stime
    print(
        f"overfit time for {epochs} epochs -> {etime/60:.4f} mins / {etime/60/60:.4f} hrs"
    )
    return model, train_loss


@click.command()
@click.option("-r", "--run", default="single_batch")
@click.option("-e", "--epochs", default=50)
@click.option("-bs", "--batch_size", default=64)
def main(run, epochs, batch_size):

    unet = Unet(dim=64)
    model = FlowMatch(unet)

    n_params = sum([p.size for p in jax.tree.leaves(nnx.state(model))])
    print(f"model parameters count: {n_params/1e6:.2f}M, ")

    # initial_learning_rate = 3e-4
    # end_learning_rate = 1e-5
    # power = 1  # setting this to 1 makes it a linear schedule

    # schedule = optax.polynomial_schedule(
    #     init_value=initial_learning_rate,
    #     end_value=end_learning_rate,
    #     power=power,
    #     transition_steps=5000,
    # )

    schedule = optax.constant_schedule(2e-4)

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(schedule, b1=0.9, b2=0.995, eps=1e-8, weight_decay=0.001),
        # optax.chain(
        #     optax.clip_by_global_norm(1.0),
        # ),
    )
    train_loader = DataLoader(
        flower_data,
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
        collate_fn=jax_collate,
    )

    sp = next(iter(train_loader))
    print(f"loaded data \n data sample: {sp['latent'].shape}")

    if run == "single_batch":
        model, loss = batch_trainer(
            epochs,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            schedule=schedule,
        )
        wandb.finish()
        print(f"single batch training ended at loss: {loss:.4f}")

    elif run == "train":
        print(f"you missed your train looop impl boy")

main()
