import PIL.Image as pillow
import torch, os, gc
import numpy as np
from typing import Tuple
from datasets import load_dataset
import torch
from diffusers import AutoencoderKL


source_data_id = "dpdl-benchmark/oxford_flowers102"
latent_id = "tensorkelechi/latent_flowers102"

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.bfloat16).to('cuda')

flower_data = load_dataset(source_data_id, split="train")
print("loaded dataset and video VAE")

def img2array(
    batch,
    target_size: Tuple[int, int] = (256, 256),
):
    img = batch['image'].resize(target_size)
    img = np.array(img, dtype=np.float32) # / 255.0
    batch["img_array"] = img # * 0.13025
    return batch


def encode_image(batch):
    with torch.no_grad():
        img_tensor = torch.tensor(batch["img_array"])[None].to(torch.bfloat16).to('cuda')
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        latents = vae.encode(img_tensor)[0]
        batch["latents"] = latents.sample()[0]
        batch["latent_shape"] = batch["latents"].shape
        del latents
        torch.cuda.empty_cache()
        gc.collect()

    return batch

print(f"start processing/downloads..")
flower_data = flower_data.map(
    img2array,
    writer_batch_size=256,
    num_proc=os.cpu_count(),
)
print(f"finished downloading and processing {len(flower_data)} flower images from {source_data_id}")

latent_data = flower_data.map(encode_image, writer_batch_size=256, num_proc=torch.cuda.device_count())
print(f"dataset preprocessing and latent encoding complete! pushed to {latent_id}")

latent_data = latent_data.remove_columns(['img_array', 'latent_shape'])

latent_data.push_to_hub(latent_id)