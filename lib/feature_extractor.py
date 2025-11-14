import os
import h5py
import openslide
from torchvision import models, transforms
import torch.nn as nn
from tqdm import tqdm
import argparse
import torch


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Remove the final fully connected layer
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(self.resnet50.children())[:-1])

    def forward(self, x):
        # shape x: (batch_size, 3, 224, 224)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output, shape: (batch_size, 2048)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from WSI patches")
    parser.add_argument(
        "--h5_dir",
        type=str,
        default="wsi_patches/BLCA/patches",
        help="Directory containing h5 files",
    )
    parser.add_argument(
        "--wsi_dir",
        type=str,
        default="wsi_files/BLCA",
        help="Directory containing WSI files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="wsi_patches/BLCA/features",
        help="Directory to save extracted features",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor().to(device)
    h5_file_path = args.h5_dir
    h5_files = os.listdir(h5_file_path)
    for h5_file in tqdm(h5_files):
        data = h5py.File(os.path.join(h5_file_path, h5_file), "r")
        wsi = openslide.open_slide(
            f"{args.wsi_dir}/{h5_file.split('/')[-1].split('.h5')[0]}.svs"
        )
        patches = []
        for coord in data["coords"][:]:
            patch = wsi.read_region(coord, 2, (224, 224)).convert("RGB")
            tensor_patch = transforms.ToTensor()(patch)
            patches.append(tensor_patch)
        patches = torch.stack(patches)  # Shape: (#patches, 3, 224, 224)
        batch_size = 32
        batches = torch.split(patches, batch_size)  # Split into batches of size 32
        features = []
        for batch in batches:
            with torch.no_grad():
                batch_features = model(
                    batch.to(device)
                )  # Shape: (32, 2048), last batch may be smaller
            features.append(batch_features.cpu())
        features = torch.cat(features, dim=0)  # Shape: (#patches, 2048)
        slide_id = h5_file.split(".h5")[0]
        torch.save(features, f"{args.output_dir}/{slide_id}.pt")
        # print(len(patches), features.shape)  # Expected output shape: (#patches, 2048)
