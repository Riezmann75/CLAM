import os
import h5py
import openslide
from torchvision import models, transforms
import torch.nn as nn
from tqdm import tqdm
import argparse
import torch

import torch
from transformers import AutoImageProcessor, ViTModel, CLIPProcessor, CLIPModel


class PLIPFeatureExtractor(nn.Module):
    def __init__(self):
        super(PLIPFeatureExtractor, self).__init__()
        self.image_processor = CLIPProcessor.from_pretrained("vinid/plip", use_fast=True)
        self.model = CLIPModel.from_pretrained("vinid/plip")

    def forward(self, x):
        # shape x: (batch_size, 3, 224, 224)
        x = self.image_processor(images=x, return_tensors="pt")
        outputs = self.model.get_image_features(**x)
        return outputs  # (batch_size, 512) shape


class UniFeatureExtractor(nn.Module):
    pass


class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(ViTFeatureExtractor, self).__init__()
        # Remove the final fully connected layer
        self.image_processor = AutoImageProcessor.from_pretrained(
            "owkin/phikon", use_fast=True
        )
        self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

    def forward(self, x):
        # shape x: (batch_size, 3, 224, 224)
        x = self.image_processor(x, return_tensors="pt")
        outputs = self.model(**x)
        x = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768) shape
        return x


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet18FeatureExtractor, self).__init__()
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            *list(self.resnet18.children())[:6]
        )  # stop before layer 3

    def forward(self, x):
        # shape x: (batch_size, 3, 224, 224)
        x = self.features(x)  # batch_size * 128 * 28 * 28
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
    parser.add_argument(
        "--patch_level",
        type=int,
        default=2,
        help="Magnification level of patches",
    )
    parser.add_argument(
        "--feature_extractor",
        type=str,
        choices=["resnet", "vit", "plip"],
        help="Feature extractor to use",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    extracted_slides = os.listdir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.feature_extractor == "vit":
        model = ViTFeatureExtractor().to(device)
    elif args.feature_extractor == "plip":
        model = PLIPFeatureExtractor().to(device)
    else:
        model = ResNet18FeatureExtractor().to(device)
    h5_file_path = args.h5_dir
    h5_files = os.listdir(h5_file_path)
    for h5_file in tqdm(h5_files):
        slide_id = h5_file.split(".h5")[0]
        if slide_id + ".pt" in extracted_slides:
            continue
        data = h5py.File(os.path.join(h5_file_path, h5_file), "r")
        wsi = openslide.open_slide(
            f"{args.wsi_dir}/{h5_file.split('/')[-1].split('.h5')[0]}.svs"
        )
        patches = []
        for coord in data["coords"][:]:
            patch = wsi.read_region(coord, args.patch_level, (224, 224)).convert("RGB")
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
                )  # Shape: (32, hidden_dim), last batch may be smaller
                assert batch_features.requires_grad == False
                features.append(batch_features.cpu())
        features = torch.cat(features, dim=0)  # Shape: (#patches, hidden_dim)
        torch.save(features, f"{args.output_dir}/{slide_id}.pt")
        print(
            len(patches), features.shape
        )  # Expected output shape: (#patches, hidden_dim)
