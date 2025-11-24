import torch.nn as nn
import torch
from pydantic import BaseModel


class PatchData(BaseModel):
    coords: list[list]
    feature_embeddings: torch.Tensor
    patch_size: int
    magnification_level: int


class HierarchicalFPN(nn.Module):
    def __init__(self, low_res_emb_dim: int, high_res_emb_dim: int):
        super(HierarchicalFPN, self).__init__()
        self.output_dim = low_res_emb_dim + high_res_emb_dim

    def forward(
        self, low_res_patches: PatchData, high_res_patches: PatchData
    ) -> torch.Tensor:
        """
        Args:
            low_res_patches (PatchData): PatchData object for low-resolution patches.
            high_res_patches (PatchData): PatchData object for high-resolution patches.

        Returns:
            torch.Tensor: Combined embeddings of shape (batch_size, num_patches, output_dim)
        """
        low_coords = torch.concat(
            [torch.tensor(coord) for coord in low_res_patches.coords], dim=0
        )  # (total_low_res_patches, 2)
        low_features = torch.concat(
            [feat for feat in low_res_patches.feature_embeddings], dim=0
        )  # (total_low_res_patches, low_res_emb_dim)
        low_patch_size = low_res_patches.patch_size
        low_mag_level = low_res_patches.magnification_level
        high_coords = torch.concat(
            [torch.tensor(coord) for coord in high_res_patches.coords], dim=0
        )  # (total_high_res_patches, 2)
        high_features = torch.concat(
            [feat for feat in high_res_patches.feature_embeddings], dim=0
        )  # (total_high_res_patches, high_res_emb_dim)
        high_patch_size = high_res_patches.patch_size
        high_mag_level = high_res_patches.magnification_level
        
        # scale factor
        scale_factor = high_mag_level // low_mag_level
        
        # find parent low-res patch for each high-res patch
        high_coords_in_low_res = high_coords // scale_factor  # (total_high_res_patches, 2)
        
        diff = high_coords_in_low_res.unsqueeze(1) - low_coords.unsqueeze(0)  # (total_high_res_patches, total_low_res_patches, 2)
        in_bound_x = (diff[:, :, 0] >= 0) & (diff[:, :, 0] < low_patch_size)
        in_bound_y = (diff[:, :, 1] >= 0) & (diff[:, :, 1] < low_patch_size)
        in_bound = in_bound_x & in_bound_y  # (total_high_res_patches, total_low_res_patches)
        
        parent_indices = in_bound.float().argmax(dim=1)  # (total_high_res_patches,) # one child - one parent assumption
        parent_low_features = low_features[parent_indices]  # (total_high_res_patches, low_res_emb_dim)
        
        # concatenate parent low-res features with high-res features
        combined_features = torch.cat([parent_low_features, high_features], dim=1)  # (total_high_res_patches, low_res_emb_dim + high_res_emb_dim)
        
        return combined_features  # (total_high_res_patches, output_dim)
