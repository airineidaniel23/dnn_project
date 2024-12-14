import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


class PairedImageDataset(Dataset):
    """
    A Dataset that returns (fenced_image, defenced_image) pairs.
    Folder structure assumption:
       train/   -> fenced images, e.g., 0.jpg, 1.jpg, ...
       target/  -> defenced images, e.g., 0.jpg, 1.jpg, ...
    """
    def __init__(self, fenced_dir, defenced_dir, transform=None):
        super().__init__()
        self.fenced_dir = fenced_dir
        self.defenced_dir = defenced_dir
        self.transform = transform
        
        # Assuming both folders have matching filenames (e.g. "0.jpg" in both)
        self.fenced_images = sorted(glob.glob(os.path.join(self.fenced_dir, "*")))
        self.defenced_images = sorted(glob.glob(os.path.join(self.defenced_dir, "*")))

        assert len(self.fenced_images) == len(self.defenced_images), \
            "Number of fenced images and defenced images must be the same."
        
    def __len__(self):
        return len(self.fenced_images)
    
    def __getitem__(self, idx):
        fenced_path = self.fenced_images[idx]
        defenced_path = self.defenced_images[idx]
        
        fenced_img = Image.open(fenced_path).convert("RGB")
        defenced_img = Image.open(defenced_path).convert("RGB")
        
        if self.transform:
            fenced_img = self.transform(fenced_img)
            defenced_img = self.transform(defenced_img)
        
        return fenced_img, defenced_img

# -----------------------------
# 2) U-Net Model Definition
# -----------------------------
class DoubleConv(nn.Module):
    """
    A helper module that performs two sequential conv3x3 + ReLU (typical U-Net block).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    A simplified U-Net for image-to-image translation with 3 input channels (RGB)
    and 3 output channels (RGB).
    """
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Downsampling path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Upsampling path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Downsampling
        for idx, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections for easier access
        skip_connections = skip_connections[::-1]
        
        # Upsampling
        for idx in range(0, len(self.ups), 2):
            transposed_conv = self.ups[idx]
            double_conv = self.ups[idx+1]
            skip_connection = skip_connections[idx//2]
            
            x = transposed_conv(x)
            
            # Concatenate skip connection
            if x.shape != skip_connection.shape:
                # Sometimes dimension rounding can differ, adapt if needed
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = double_conv(x)
        
        return self.final_conv(x)

# -----------------------------
# 3) Training Script
# -----------------------------
def train_unet(
    fenced_dir,
    defenced_dir,
    batch_size=2,
    epochs=10,
    lr=1e-4,
    device="cuda"
):
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize for memory reasons; adapt as needed
        transforms.ToTensor(),
    ])
    
    # Dataset & DataLoader
    dataset = PairedImageDataset(fenced_dir, defenced_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Initialize model, loss, optimizer
    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.L1Loss()  # L1 or MSE. You can also try a perceptual loss or combo of losses.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for fenced_batch, defenced_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            fenced_batch = fenced_batch.to(device)
            defenced_batch = defenced_batch.to(device)
            
            optimizer.zero_grad()
            output = model(fenced_batch)
            
            loss = criterion(output, defenced_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * fenced_batch.size(0)
        
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")
    
    return model

# -----------------------------
# 4) Inference / Testing
# -----------------------------
def inference(model, input_image_path, device="cuda"):
    """
    Run the model on a single fenced image and return defenced result.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    model.eval()
    with torch.no_grad():
        img = Image.open(input_image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        output = model(img_tensor)
        output = torch.clamp(output, 0, 1)  # clamp to [0,1]
        
        # Convert back to PIL image
        output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
        return output_image

# -----------------------------
# 5) Usage Example
# -----------------------------
if __name__ == "__main__":
    # Example usage
    fenced_dir = "input"   # Folder of fenced images
    defenced_dir = "target"  # Folder of defenced images

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train
    trained_model = train_unet(
        fenced_dir=fenced_dir,
        defenced_dir=defenced_dir,
        batch_size=4,
        epochs=10,       # Increase for better results
        lr=1e-4,
        device=device
    )
    
    # Save model weights
    torch.save(trained_model.state_dict(), "mma_fence_removal_unet.pth")

    # Inference on a new fenced image
    test_image_path = "test.jpg"
    result_image = inference(trained_model, test_image_path, device=device)
    result_image.save("defenced_output.jpg")
    print("Saved defenced image to defenced_output.jpg")
