import os
import einops
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
from torchvision import transforms


# NOTE: Transform input images
img_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# NOTE: Transform corresponding masks
mask_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(), # float in [0, 1]
    transforms.Lambda(lambda x: (x > 0.5).float()) # binarize
])


def plot_train_vs_val_loss(train_losses, val_losses, save_path, file_desc):
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"train_vs_val_loss_{file_desc}.png"))
    plt.show()


def plot_test_sample(model, sample_path, device, save_path, file_desc):
    img = img_transform(Image.open(sample_path))                       # (1,H,W)
    x = einops.rearrange(img, "c h w -> 1 c h w").to(device)           # (1,1,H,W)
    img_np = img.squeeze().cpu().numpy()

    with torch.no_grad():
        pred = torch.sigmoid(model(x))
    pred_np = pred.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img_np, cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")

    ax[1].imshow(img_np, cmap="gray")
    ax[1].contour(pred_np, levels=[0.5], colors="red", linewidths=0.5)
    ax[1].legend([Line2D([0], [0], color="red", lw=0.5)], ["Prediction"], loc="lower right")
    ax[1].set_title("Overlay"); ax[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"test_sample_prediction_{file_desc}.png"))
    plt.close()