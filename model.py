import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import plot_train_vs_val_loss


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1): # 128x128
        super().__init__()
        # Encoder - level 1
        self.enc_l1_conv1 = nn.Conv2d(in_ch, out_channels=32, kernel_size=3, padding=1)
        self.enc_l1_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x64
        # Encoder - level 2
        self.enc_l2_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.enc_l2_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32
        # Encoder - level 3
        self.enc_l3_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.enc_l3_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        # Decoder - level 3
        self.dec_l3_deconv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2) # 32x32
        self.dec_l3_conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.dec_l3_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # Decoder - level 2
        self.dec_l2_deconv = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec_l2_conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.dec_l2_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # Decoder - level 1
        self.dec_l1_deconv = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.dec_l1_conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.dec_l1_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # Final output layer
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=out_ch, kernel_size=1)


    def forward(self, x):
        # Encoder - level 1
        enc_l1_c1 = F.relu(self.enc_l1_conv1(x))
        enc_l1_c2 = F.relu(self.enc_l1_conv2(enc_l1_c1)) # save for skip connection
        enc_l1_p = self.pool1(enc_l1_c2)
        # Encoder - level 2
        enc_l2_c1 = F.relu(self.enc_l2_conv1(enc_l1_p))
        enc_l2_c2 = F.relu(self.enc_l2_conv2(enc_l2_c1)) # save for skip connection
        enc_l2_p = self.pool2(enc_l2_c2)
        # Encoder - level 3
        enc_l3_c1 = F.relu(self.enc_l3_conv1(enc_l2_p))
        enc_l3_c2 = F.relu(self.enc_l3_conv2(enc_l3_c1)) # save for skip connection
        enc_l3_p = self.pool3(enc_l3_c2)

        # Bottleneck
        bottleneck_c1 = F.relu(self.bottleneck_conv1(enc_l3_p))
        bottleneck_c2 = F.relu(self.bottleneck_conv2(bottleneck_c1))

        # Decoder - level 3
        dec_l3_u = self.dec_l3_deconv(bottleneck_c2)        
        dec_l3_cat = torch.cat((dec_l3_u, enc_l3_c2), dim=1) # skip connection
        dec_l3_c1 = F.relu(self.dec_l3_conv1(dec_l3_cat))        
        dec_l3_c2 = F.relu(self.dec_l3_conv2(dec_l3_c1))
        # Decoder - level 2
        dec_l2_u = self.dec_l2_deconv(dec_l3_c2)
        dec_l2_cat = torch.cat((dec_l2_u, enc_l2_c2), dim=1) # skip connection
        dec_l2_c1 = F.relu(self.dec_l2_conv1(dec_l2_cat))        
        dec_l2_c2 = F.relu(self.dec_l2_conv2(dec_l2_c1))
        # Decoder - level 1
        dec_l1_u = self.dec_l1_deconv(dec_l2_c2)        
        dec_l1_cat = torch.cat((dec_l1_u, enc_l1_c2), dim=1) # skip connection
        dec_l1_c1 = F.relu(self.dec_l1_conv1(dec_l1_cat))        
        dec_l1_c2 = F.relu(self.dec_l1_conv2(dec_l1_c1))
        
        # Final output layer
        out = self.final_conv(dec_l1_c2)

        return out  # logits



def train(model, train_loader, val_loader, loss_fn, optimizer, epochs,
          model_path, save_path, file_desc, device=torch.device("cpu")):
    """Train the UNet model and validate after each epoch."""
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # ------- Training -------
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            # 1. Forward pass
            output_logits = model(images)
            # 2. Compute loss
            loss = loss_fn(output_logits, masks)
            # 3. Backpropagation (compute gradients)
            loss.backward()
            # 4. Update model parameters
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ------- Validation -------
        epoch_val_loss = evaluate(model, val_loader, loss_fn, device=device)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], "f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        # ------- wandb log -------
        wandb.log({"epoch": epoch+1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss}) # type: ignore

    torch.save(model.state_dict(), model_path)
    plot_train_vs_val_loss(train_losses, val_losses, save_path=save_path, file_desc=file_desc)


def evaluate(model, dataloader, loss_fn, device=torch.device("cpu")):
    """Evaluate the model on the given dataloader."""
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            test_loss += loss.item() * images.size(0)
    return test_loss / len(dataloader.dataset)