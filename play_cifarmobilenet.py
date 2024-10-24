import torch
from torch import nn
from torch.nn import functional as func_nn
from einops import rearrange
from torchvision import models
from datasets import load_dataset
import os, cv2, gc, wandb
import numpy as np
from torch.amp import GradScaler
from tqdm.auto import tqdm
from safetensors.torch import save_model
from torch.utils.data import IterableDataset, DataLoader


cifar_id = "uoft-cs/cifar10"
split_size = 1000

data = load_dataset(cifar_id, split="train", trust_remote_code=True, streaming=True)
data = data.take(split_size)


def read_image(img, img_size=32):
    img = np.array(img)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    return img


class Cifar(IterableDataset):
    def __init__(self, dataset=data):
        self.dataset = dataset

    def __len__(self):
        return split_size

    def __iter__(self):
        for sample in self.dataset:
            image = read_image(sample["image"])
            label = sample["label"]
            
            image = torch.as_tensor(image, dtype=torch.float32)
            label = torch.as_tensor(label, dtype=torch.long)

            yield image, label

train_data = Cifar()
train_loader = DataLoader(train_data, batch_size=32)


# main model network
class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()

        # convolutional layer/block

        self.convnet = models.mobilenet_v2(pretrained=True) 
        num_ftrs = self.convnet.classifier[1].in_features 

        self.convnet.classifier[1] = nn.Linear(num_ftrs, 256)

        # fully connected layer for classification
        self.fc_linear = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),  # actvation layer
            nn.Linear(128, 10),
        )
    
    def forward(self, x_img):
        x = rearrange(x_img, 'b h w c -> b c h w')
        
        x = self.convnet(x)
        x = self.fc_linear(x)
        
        return x


mobilenet_classifier = ToyNet()

out = mobilenet_classifier(torch.randn(1, 32, 32, 3))

print(out.shape)


# # Model parameter count
def count_params(model: torch.nn.Module):
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return p_count

def clearmem():
    torch.cuda.empty_cache()
    gc.collect()


def w_logger(key: str, model):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    train_run = wandb.init(project="play_cifar", name="func_model1")
    wandb.watch(model)


w_logger()

optimizer = torch.optim.AdamW(mobilenet_classifier.parameters(), lr=1e-4)

# basic training loop
def trainer(
    model, train_loader, epochs, config, optimizer,
):
    scaler = GradScaler(device='cuda')
    device = torch.device("cuda")
    
    model.train()
    train_loss = 0.0

    for epoch in tqdm(range(epochs)):
        clearmem()
        optimizer.zero_grad()

        print(f"Training epoch {epoch+1}")

        for x, (image, label) in tqdm(enumerate(train_loader)):
            image = image.to(device)
            label = label.to(device)

            # Mixed precision training
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                
                output = model(image)
                
                train_loss = func_nn.cross_entropy(output, label.long())
                print(f'step {x}: loss {train_loss.item():.4f}')
                
                train_loss = train_loss / config.grad_acc_step  # Normalize the loss

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(train_loss).backward()

            if (x + 1) % config.grad_acc_step == 0:
                # Unscales the gradients of optimizer's assigned params in-place

                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                optimizer.zero_grad()

            wandb.log({"loss": train_loss})

        print(f"Epoch {epoch} of {epochs}, train_loss: {train_loss.item():.4f}")

        print(f"Epoch @ {epoch} complete!")

    print(f"End metrics for run of {epochs}, train_loss: {train_loss.item():.4f}")

    safe_tensorfile = save_model(model, config.safetensor_file)

trainer()

# torch.cuda.empty_cache()
# gc.collect()

# Ciao
