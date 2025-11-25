import os

from create_patches_fp import seg_and_patch
import argparse

from wsi_file_sorter import sort_wsi_files


class PatchExtractor:
    def __init__(
        self,
        source: str,
        save_dir: str,
        patch_size: int,
        patch_level: int = None,
        patch: bool = True,
        seg: bool = True,
        stitch: bool = True,
    ):
        self.source = source
        self.save_dir = save_dir
        self.patch_size = patch_size
        self.patch_level = patch_level
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.patch = patch
        self.seg = seg
        self.stitch = stitch

    def _get_directories(self):
        if self.patch:
            patch_dir = os.path.join(self.save_dir, "patches")
            if not os.path.exists(patch_dir):
                os.makedirs(patch_dir, exist_ok=True)
        if self.seg:
            seg_dir = os.path.join(self.save_dir, "segmented")
            if not os.path.exists(seg_dir):
                os.makedirs(seg_dir, exist_ok=True)
        if self.stitch:
            stitch_dir = os.path.join(self.save_dir, "stitched")
            if not os.path.exists(stitch_dir):
                os.makedirs(stitch_dir, exist_ok=True)
        return {
            "source": self.source,
            "save_dir": self.save_dir,
            "patch_save_dir": patch_dir,
            "mask_save_dir": seg_dir,
            "stitch_save_dir": stitch_dir,
        }

    def extract_patches(self, slides=None):
        seg_times, patch_times = seg_and_patch(
            **self._get_directories(),
            patch_size=self.patch_size,
            step_size=self.patch_size,
            seg=self.seg,
            use_default_params=False,
            save_mask=True,
            stitch=self.stitch,
            patch_level=self.patch_level,
            patch=self.patch,
            slides=slides,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patches from WSI files")
    parser.add_argument(
        "--source",
        type=str,
        default="wsi_files/BLCA",
        required=True,
        help="Directory containing WSI files",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="wsi_patches/BLCA",
        required=True,
        help="Directory to save extracted patches",
    )
    parser.add_argument(
        "--patch_level",
        type=int,
        default=0,
        help="Magnification level of patches",
    )
    parser.add_argument(
        "--patch",
        action="store_true",
        help="Whether to extract patches",
    )
    parser.add_argument(
        "--seg",
        action="store_true",
        help="Whether to segment tissue regions",
    )
    parser.add_argument(
        "--stitch",
        action="store_true",
        help="Whether to stitch masks back to WSI size",
    )
    parser.add_argument(
        "--target_magnification",
        type=float,
        default=2.5,
        help="Target magnification level for sorting WSI files",
    )

    args = parser.parse_args()

    magnification_map = sort_wsi_files(
        args.source, args.target_magnification, args.patch_level
    )
    for patch_size, wsi_files in magnification_map.items():
        args.patch_size = patch_size
        patch_extractor = PatchExtractor(
            source=args.source,
            save_dir=args.save_dir,
            patch_size=patch_size,
            patch_level=args.patch_level,
            patch=args.patch,
            seg=args.seg,
            stitch=args.stitch,
        )
        patch_extractor.extract_patches(slides=wsi_files)
