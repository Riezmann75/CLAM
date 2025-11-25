import math
import os
import time
import h5py
import numpy as np
import openslide
from torchvision import models, transforms
import torch.nn as nn
from tqdm import tqdm
import argparse
import torch

import torch
from transformers import AutoImageProcessor, ViTModel, CLIPProcessor, CLIPModel
from simclr import load_model
from torch.utils.data import DataLoader, Dataset


class WSIPatchDataset(Dataset):
    def __init__(self, slide_path, coords, patch_level, patch_size, transform=None):
        """
        Args:
            data_list (list): List of tuples containing (patient_id, patches, coordinates, clinical_outcomes, mask)
        """
        self.slide_path = slide_path
        self.coords = coords
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.transform = transform
        self.wsi = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self.wsi is None:
            import openslide

            self.wsi = openslide.OpenSlide(self.slide_path)

        coord = self.coords[idx]

        # Read region
        patch = self.wsi.read_region(
            location=coord,
            level=self.patch_level,
            size=(self.patch_size, self.patch_size),
        ).convert("RGB")

        if self.transform:
            patch = self.transform(patch)

        return patch


class PLIPFeatureExtractor(nn.Module):
    def __init__(self):
        super(PLIPFeatureExtractor, self).__init__()
        self.image_processor = CLIPProcessor.from_pretrained(
            "vinid/plip", use_fast=True
        )
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


class SimCLRFeatureExtractor(nn.Module):
    def __init__(self, simclr_path, device=None):
        super(SimCLRFeatureExtractor, self).__init__()
        self.simclr_model = load_model(simclr_path, device=device)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )

    def forward(self, x):
        # shape x: (batch_size, 3, 224, 224)
        self.simclr_model.eval()
        with torch.no_grad():
            features = self.simclr_model(x.to(self.device))  # shape (batch_size, 512)

        return features


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
        default=0,
        help="Level of the patches to extract",
    )
    parser.add_argument(
        "--target_patch_size",
        type=int,
        default=224,
        help="Size of the patches to extract",
    )
    parser.add_argument(
        "--feature_extractor",
        type=str,
        choices=["resnet", "vit", "plip", "simclr"],
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
    elif args.feature_extractor == "simclr":
        simclr_model_path = "pretrained_encoders/tenpercent_resnet18.ckpt"
        model = SimCLRFeatureExtractor(simclr_model_path, device=device).to(device)
    else:
        model = ResNet18FeatureExtractor().to(device)
    h5_file_path = args.h5_dir
    h5_files = os.listdir(h5_file_path)
    print(extracted_slides)
    for h5_file in tqdm(h5_files):
        slide_id = h5_file.split(".h5")[0]
        if slide_id + ".pt" in extracted_slides:
            continue
        start = time.time()
        data = h5py.File(os.path.join(h5_file_path, h5_file), "r")
        end = time.time()
        print(f"Time to read h5 file: {end - start} seconds")
        start = time.time()
        wsi = openslide.open_slide(
            f"{args.wsi_dir}/{h5_file.split('/')[-1].split('.h5')[0]}.svs"
        )
        end = time.time()
        print(f"Time to open WSI file: {end - start} seconds")
        patches = []

        patch_size = None
        for i in range(len(data["coords"][:]) - 1):
            current_coord = data["coords"][:][i]
            next_coord = data["coords"][:][i + 1]
            if next_coord[0] == current_coord[0]:
                patch_size = next_coord[1] - current_coord[1]
                break
            elif next_coord[1] == current_coord[1]:
                patch_size = next_coord[0] - current_coord[0]
                break
            else:
                continue
        assert patch_size is not None, "Could not determine patch size from coordinates"

        batch_size = 32
        print(f"Total number of patches: {len(data['coords'][:])}")
        batches = np.array_split(
            data["coords"][:], math.ceil(len(data["coords"][:]) / batch_size)
        )
        features = []
        resizer = transforms.Resize((args.target_patch_size, args.target_patch_size))
        to_tensor = transforms.ToTensor()
        patch_transforms = transforms.Compose(
            [
                transforms.Resize((args.target_patch_size, args.target_patch_size)),
                transforms.ToTensor(),
            ]
        )
        dataset = WSIPatchDataset(
            slide_path=f"{args.wsi_dir}/{h5_file.split('/')[-1].split('.h5')[0]}.svs",
            coords=data["coords"][:],
            patch_level=args.patch_level,
            patch_size=patch_size,
            transform=patch_transforms,
        )
        loader = DataLoader(dataset, batch_size=32, num_workers=8, pin_memory=True)
        features = []
        for batch_patches in tqdm(loader, desc=f"Processing {slide_id}"):
            with torch.no_grad():
                batch_features = model(batch_patches.to(device))
                features.append(batch_features.cpu())

        features = torch.cat(features, dim=0)
        torch.save(features, f"{args.output_dir}/{slide_id}.pt")
