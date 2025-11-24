import os
import openslide


def move_file_to_dir(file_path, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    file_name = os.path.basename(file_path)
    target_path = os.path.join(target_dir, file_name)
    os.rename(file_path, target_path)
    return target_path


def sort_wsi_files(wsi_dir, target_magnification):
    wsi_files = [f for f in os.listdir(wsi_dir) if f.endswith(".svs")]
    magnification_map = {}
    for wsi_file in wsi_files:
        wsi = openslide.open_slide(os.path.join(wsi_dir, wsi_file))
        objective_power = int(
            wsi.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        )
        scale_factor = objective_power / target_magnification
        patch_size = int(224 * scale_factor)
        if patch_size not in magnification_map:
            magnification_map[patch_size] = []
        magnification_map[patch_size].append(wsi_file)
    return magnification_map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sort WSI files by magnification")
    parser.add_argument(
        "--wsi_dir",
        type=str,
        required=True,
        help="Directory containing WSI files",
    )
    parser.add_argument(
        "--target_magnification",
        type=float,
        default=2.5,
        help="Target magnification level to sort WSI files",
    )
    args = parser.parse_args()

    print(sort_wsi_files(args.wsi_dir, args.target_magnification))

    # sample command to run:
    # python lib/wsi_file_sorter.py --wsi_dir ./wsi_files/BLCA --target_magnification 16
