import os
import glob
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

#########################
# 1) U-Net Definition
#########################

class DoubleConv(nn.Module):
    """
    Two successive 3x3 convolutions, each followed by ReLU.
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
    A simplified U-Net for image-to-image translation.
    Input:  3-channel RGB
    Output: 3-channel RGB
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
        
        # Down
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        
        # Up
        for idx in range(0, len(self.ups), 2):
            transposed_conv = self.ups[idx]
            double_conv = self.ups[idx+1]
            skip_connection = skip_connections[idx//2]
            
            x = transposed_conv(x)
            # If shapes mismatch slightly, resize x
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = double_conv(x)
        
        return self.final_conv(x)

#########################
# 2) Inference Function
#########################

def fence_removal_inference(model, input_image_path, device="cuda"):
    """
    - Load image from input_image_path
    - Resize to 256x256
    - Run model inference
    - Return defenced PIL image, upscaled to 640x360
    """
    transform_256 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    model.eval()
    with torch.no_grad():
        img = Image.open(input_image_path).convert("RGB")
        img_tensor = transform_256(img).unsqueeze(0).to(device)
        
        output = model(img_tensor)
        output = torch.clamp(output, 0, 1)  # clamp to [0,1]
        
        # Convert tensor to PIL (256x256)
        output_256 = transforms.ToPILImage()(output.squeeze(0).cpu())
        
        # Resize from 256x256 to 640x360
        output_640x360 = output_256.resize((640, 360), Image.BICUBIC)
        return output_640x360

#########################
# 3) Main Inference Script
#########################

def infer_on_folder(
    model_path, 
    fenced_folder, 
    output_folder="testFullVideoOutput", 
    device="cuda"
):
    """
    Loads the UNet model from `model_path`, runs inference on every image in `fenced_folder`, 
    and saves the results to `output_folder` at 640x360 resolution.
    """
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Device
    device = device if torch.cuda.is_available() else "cpu"
    
    # Initialize model & load weights
    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Gather input images
    image_paths = sorted(
        glob.glob(os.path.join(fenced_folder, "*.*"))  # .jpg, .png, etc.
    )
    
    for img_path in image_paths:
        file_name = os.path.basename(img_path)
        output_path = os.path.join(output_folder, file_name)
        
        # Run inference
        result_image = fence_removal_inference(model, img_path, device=device)
        
        # Save result
        result_image.save(output_path)
        print(f"Saved: {output_path}")

#########################
# 4) Example Usage
#########################

if __name__ == "__main__":
    # Path to the .pth model
    model_path = "mma_fence_removal_unet.pth"
    
    # Folder containing fenced images
    fenced_folder = "fullVideoTestFenced"
    
    # Where to save results
    output_folder = "testFullVideoOutput"
    
    # Run inference
    infer_on_folder(model_path, fenced_folder, output_folder=output_folder, device="cuda")
    print("Inference complete.")
