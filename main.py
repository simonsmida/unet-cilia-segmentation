import os
import wandb
import torch
from PIL import Image

from model import UNet, train, evaluate
from dataset import CiliaDataset, make_loaders
from utils import img_transform, mask_transform, plot_test_sample

# pyright: reportAttributeAccessIssue=false

# Configuration
COMMENT = "skip" # using skip connections
EPOCHS = 40
BATCH_SIZE = 4
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = "./"
DATA_PATH   = os.path.join(ROOT, "data")
VIS_PATH    = os.path.join(ROOT, "visuals")
MODELS_PATH = os.path.join(ROOT, "models")


def main():

    print(f"Using device: {DEVICE}")
    wandb.init( 
        project="cilia-unet",    
        config={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "loss": "BCEWithLogitsLoss", "architecture": "UNet"}
    )

    # Prepare dataset and dataloaders
    dataset = CiliaDataset(DATA_PATH, transform=img_transform, mask_transform=mask_transform)
    train_loader, val_loader, test_loader = make_loaders(dataset, _batch_size=BATCH_SIZE)

    # Try loading existing model, else train new
    model = UNet(in_ch=1, out_ch=1).to(DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    file_desc = f"e{EPOCHS}_{BATCH_SIZE}_{LR}_{COMMENT}"
    model_path = os.path.join(MODELS_PATH, f"unet_cilia_{file_desc}.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        print(f"Loaded existing model from '{model_path}'")
    else:
        print("Training a new model...")
        train(model, train_loader, val_loader, loss_fn, optimizer, epochs=EPOCHS, 
              model_path=model_path, save_path=VIS_PATH, file_desc=file_desc, device=DEVICE)

    # Evaluate on test set
    test_loss = evaluate(model, test_loader, loss_fn, device=DEVICE)
    print(f"Test Loss: {test_loss:.4f}")
    wandb.log({"test_loss": test_loss})

    # NOTE: TEST SAMPLE outside of dataset
    plot_test_sample(model, 
                     sample_path=os.path.join(DATA_PATH, "cilia_test_sample.png"), 
                     device=DEVICE, save_path=VIS_PATH, file_desc=file_desc)
    
    wandb.finish()    


if __name__ == "__main__":
    main()