from tqdm import tqdm
import pandas as pd
import os
import requests
import re

save_dir = "./wsi_files/BLCA"


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


txt_file_path = "gdc_manifest.2025-11-08.140931.txt"
manifest_file = pd.read_csv(txt_file_path, sep="\t")
open("wsi_files/BLCA/saved_ids.txt", "w").close()
for id in tqdm(manifest_file["id"]):
    download_svs(id, save_dir)
    with open("wsi_files/BLCA/saved_ids.txt", "a") as file:
        file.write(id + "\n")
