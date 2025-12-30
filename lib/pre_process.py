import os
import pdb
from typing import Literal, Optional
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class PatientDataset(Dataset):
    def __init__(
        self,
        extracted_dir: str,
        slide_ids: list[str],
        h5_files: list[str],
        X,
        y,
        mode: Literal["train", "validation"],
        max_patch_per_patient: Optional[int] = None,
    ):
        self.extracted_dir = extracted_dir
        self.slide_ids = slide_ids
        self.h5_files = h5_files
        self.X = X
        self.y = y
        self.patches = []
        self.clinical_cols = ["survival_months", "censorship"]
        self.feature_cols = [
            col for col in self.X.columns if col not in self.clinical_cols
        ]
        self.feature_cols.remove("slide_id")
        self.max_patch_per_patient = max_patch_per_patient
        self.mode = mode
        self.h5_files_paths = []
        self.features_paths = []
        for h5_file in h5_files:
            assert os.path.exists(h5_file), f"H5 file {h5_file} does not exist."
        for slide_id in slide_ids:
            # load tensor from extracted features
            features_dir = os.path.join(self.extracted_dir, "features")
            coords_dir = os.path.join(self.extracted_dir, "coords")
            self.h5_files_paths.append(os.path.join(coords_dir, f"{slide_id}.h5"))
            self.features_paths.append(os.path.join(features_dir, f"{slide_id}.pt"))

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        slide_id = slide_id + ".svs"
        patient = self.X[self.X["slide_id"] == slide_id].iloc[0]
        patient = patient[self.feature_cols]
        patient = torch.tensor(patient.values.astype(float))

        if self.max_patch_per_patient is None:
            features_path = self.features_paths[idx]
            patch_features = torch.load(features_path)
            coordinates = h5py.File(self.h5_files_paths[idx], "r")
            coordinates = coordinates["coords"][:]
            return (patient, patch_features, coordinates), torch.tensor(
                self.y.iloc[idx].values.astype(float)
            )
        else:
            features_path = self.features_paths[idx]
            patient_patch_features = torch.load(features_path)
            patient_coordinates = h5py.File(self.h5_files_paths[idx], "r")
            patient_coordinates = patient_coordinates["coords"][:]
            if self.mode == "train":
                num_patch = patient_patch_features.shape[0]
                if num_patch > self.max_patch_per_patient:
                    indices = torch.randperm(num_patch)[: self.max_patch_per_patient]
                    indices, _ = torch.sort(indices)
                    patch_features = patient_patch_features[indices]
                    coordinates = []
                    coordinates.append(patient_coordinates[indices])
                    return (patient, patch_features, coordinates), torch.tensor(
                        self.y.iloc[idx].values.astype(float)
                    )
                else:
                    return (
                        patient,
                        patient_patch_features,
                        patient_coordinates,
                    ), torch.tensor(self.y.iloc[idx].values.astype(float))
            if self.mode == "validation":
                patch_features = patient_patch_features[: self.max_patch_per_patient]
                coordinates = patient_coordinates[: self.max_patch_per_patient]
                return (patient, patch_features, coordinates), torch.tensor(
                    self.y.iloc[idx].values.astype(float)
                )


def collate_fn(batch):
    patients, patches, coordinates, clinical_outcomes = [], [], [], []
    for (patient, image_patches, coords), clinical_outcome in batch:
        patients.append(patient)
        patches.append(image_patches)
        coordinates.append(coords)
        clinical_outcomes.append(clinical_outcome)
    max_num_patches = max([patch.shape[0] for patch in patches])
    mask = torch.ones(len(patches), max_num_patches, dtype=torch.bool)
    for i, patch in enumerate(patches):
        mask[i, : patch.shape[0]] = 0

    patients = torch.stack(patients)
    patches = pad_sequence(patches, batch_first=True)
    # shape: batch * max_num_patches * 2
    coordinates = pad_sequence(
        [torch.from_numpy(coords) for coords in coordinates], batch_first=True
    )

    clinical_outcomes = torch.stack(clinical_outcomes)
    return patients, patches, coordinates, clinical_outcomes, mask


def load_dataset(
    clean_csv_path: str,
    extracted_dir: str,
    batch_size=4,
    num_workers=2,
    max_patch_per_patient: Optional[int] = None,
):
    df = pd.read_csv(clean_csv_path)
    h5_files = os.listdir(os.path.join(extracted_dir, "coords"))
    df_filtered = df.drop(columns=["site", "oncotree_code"])
    train_df = df[df["train"] == 1]
    test_df = df[df["train"] == 0]
    train_df = train_df.drop(columns=["train"])
    test_df = test_df.drop(columns=["train"])
    # get the patients in file_ids
    wsi_path = "./wsi_files/BLCA/"
    slide_ids = [
        file_id for file_id in os.listdir(wsi_path) if file_id.endswith(".svs")
    ]
    df_filtered = df_filtered.drop(columns=["train"])
    df_filtered = df_filtered[df_filtered["slide_id"].isin(slide_ids)]
    clinical_cols = ["survival_months", "censorship"]
    feature_cols = [col for col in df_filtered.columns if col not in clinical_cols]
    feature_cols.remove("case_id")
    categorical_cols = ["is_female"]
    numeric_cols = []
    for col in feature_cols:
        if col not in categorical_cols:
            numeric_cols.append(col)

    # X = df_filtered[feature_cols]
    # y = df_filtered[clinical_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[clinical_cols]
    X_test = test_df[feature_cols]
    y_test = test_df[clinical_cols]

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42, stratify=y["censorship"]
    # )

    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train["censorship"]
    )

    train_slide_ids = X_train["slide_id"].to_list()
    train_slide_ids = [slide_id.split(".svs")[0] for slide_id in train_slide_ids]
    test_slide_ids = X_test["slide_id"].to_list()
    test_slide_ids = [slide_id.split(".svs")[0] for slide_id in test_slide_ids]
    validate_slide_ids = X_validate["slide_id"].to_list()
    validate_slide_ids = [slide_id.split(".svs")[0] for slide_id in validate_slide_ids]

    train_h5_files = [
        os.path.join(extracted_dir, "coords", h5_file)
        for h5_file in h5_files
        if h5_file in train_slide_ids
    ]
    test_h5_files = [
        os.path.join(extracted_dir, "coords", h5_file)
        for h5_file in h5_files
        if h5_file in test_slide_ids
    ]
    validate_h5_files = [
        os.path.join(extracted_dir, "coords", h5_file)
        for h5_file in h5_files
        if h5_file in validate_slide_ids
    ]

    train_dataset = PatientDataset(
        extracted_dir=extracted_dir,
        slide_ids=train_slide_ids,
        h5_files=train_h5_files,
        X=X_train.reset_index(drop=True),
        y=y_train.reset_index(drop=True),
        max_patch_per_patient=max_patch_per_patient,
        mode="train",
    )

    test_dataset = PatientDataset(
        extracted_dir=extracted_dir,
        slide_ids=test_slide_ids,
        h5_files=test_h5_files,
        X=X_test.reset_index(drop=True),
        y=y_test.reset_index(drop=True),
        max_patch_per_patient=max_patch_per_patient,
        mode="validation",
    )

    validate_dataset = PatientDataset(
        extracted_dir=extracted_dir,
        slide_ids=validate_slide_ids,
        h5_files=validate_h5_files,
        X=X_validate.reset_index(drop=True),
        y=y_validate.reset_index(drop=True),
        max_patch_per_patient=max_patch_per_patient,
        mode="validation",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    validate_loader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "validate_dataset": validate_dataset,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "validate_loader": validate_loader,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "filtered_df": df_filtered,
    }
