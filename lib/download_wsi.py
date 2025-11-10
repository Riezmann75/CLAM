from tqdm import tqdm
import pandas as pd
import os
import requests
import re
import argparse

save_dir = "../wsi_files/BLCA"


def download_svs(file_id, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    data_endpt = "https://api.gdc.cancer.gov/data/{}".format(file_id)

    response = requests.get(
        data_endpt, headers={"Content-Type": "application/octet-stream"}
    )

    # The file name can be found in the header within the Content-Disposition key.
    response_head_cd = response.headers["Content-Disposition"]

    file_name = re.findall("filename=(.+)", response_head_cd)[0]

    with open(os.path.join(save_dir, file_name), "wb") as output_file:
        output_file.write(response.content)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download WSI files from GDC")
    parser.add_argument("--save_dir", type=str, default="../wsi_files/BLCA", help="Directory to save WSI files")
    parser.add_argument("--cleaned_csv_path", type=str, default="dataset_csv/tcga_blca_all_clean.csv", help="Path to cleaned CSV file")
    parser.add_argument("--manifest_txt_path", type=str, default="dataset_csv/gdc_manifest.2025-11-08.140931.txt", help="Path to GDC manifest txt file")
    
    args = parser.parse_args()
    save_dir = args.save_dir
    cleaned_csv_path = args.cleaned_csv_path
    manifest_txt_path = args.manifest_txt_path

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(cleaned_csv_path)
    open(os.path.join(save_dir, "saved_ids.txt"), "w").close()
    manifest_file = pd.read_csv(manifest_txt_path, sep="\t")
    slide_ids = df["slide_id"]
    selected_manifest = manifest_file[
        manifest_file["filename"].isin(slide_ids)
    ]  # filter manifest to only include files in slide_ids
    for row in tqdm(selected_manifest.itertuples(), desc="Downloading WSI files"):
        file_id = row.id
        file_name = row.filename
        try:
            download_svs(file_id, save_dir)
            with open(os.path.join(save_dir, "saved_ids.txt"), "a") as f:
                f.write(f"{file_name}\n")
        except Exception as e:
            print(f"Failed to download {file_name} with error {e}")
