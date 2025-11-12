import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class PatientDataset(Dataset):
    def __init__(self, case_ids: list[str], h5_files: list[str], X, y):
        self.case_ids = case_ids
        self.h5_files = h5_files
        self.X = X
        self.y = y
        self.patches = []
        self.clinical_cols = ["survival_months", "censorship"]
        self.feature_cols = [
            col for col in self.X.columns if col not in self.clinical_cols
        ]
        self.feature_cols.remove("case_id")
        self.patch_features = []
        for h5_file in h5_files:
            assert os.path.exists(h5_file), f"H5 file {h5_file} does not exist."
        for case_id in case_ids:
            # load tensor from extracted features
            extracted_path = f"wsi_patches/BLCA/features/{case_id}.pt"
            features = torch.load(extracted_path)
            self.patch_features.append(features)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        patch_features = self.patch_features[idx]
        case_id = self.case_ids[idx]
        patient = self.X[self.X["case_id"] == case_id].iloc[0]
        patient = patient[self.feature_cols]
        patient = torch.tensor(patient.values.astype(float))
        return (patient, patch_features), torch.tensor(
            self.y.iloc[idx].values.astype(float)
        )


def collate_fn(batch):
    patients, patches, clinical_outcomes = [], [], []
    for (patient, image_patches), clinical_outcome in batch:
        patients.append(patient)
        patches.append(image_patches)
        clinical_outcomes.append(clinical_outcome)
    max_num_patches = max([patch.shape[0] for patch in patches])
    mask = torch.zeros(len(patches), max_num_patches, dtype=torch.bool)
    for i, patch in enumerate(patches):
        mask[i, : patch.shape[0]] = True

    patients = torch.stack(patients)
    patches = pad_sequence(patches, batch_first=True)

    clinical_outcomes = torch.stack(clinical_outcomes)
    return patients, patches, clinical_outcomes, mask


def load_dataset(clean_csv_path: str, h5_dir: str, h5_files: list[str]):
    df = pd.read_csv(clean_csv_path)
    df_filtered = df.drop(columns=["site", "oncotree_code", "train"])
    # get the patients in file_ids
    wsi_path = "./wsi_files/BLCA/"
    slide_ids = [
        file_id for file_id in os.listdir(wsi_path) if file_id.endswith(".svs")
    ]
    case_ids = [slide_id.split("-01Z")[0] for slide_id in slide_ids]
    case_ids = list(set(case_ids))
    df_filtered = df_filtered[df_filtered["slide_id"].isin(slide_ids)]
    clinical_cols = ["survival_months", "censorship"]
    feature_cols = [col for col in df_filtered.columns if col not in clinical_cols]
    feature_cols.remove("slide_id")
    categorical_cols = ["is_female"]
    numeric_cols = []
    for col in feature_cols:
        if col not in categorical_cols:
            numeric_cols.append(col)

    X = df_filtered[feature_cols]
    y = df_filtered[clinical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    train_case_ids = X_train["case_id"].to_list()
    test_case_ids = X_test["case_id"].to_list()
    validate_case_ids = X_validate["case_id"].to_list()
    train_h5_files = [
        os.path.join(h5_dir, h5_file)
        for h5_file in h5_files
        if h5_file.split("-01Z")[0] in train_case_ids
    ]
    test_h5_files = [
        os.path.join(h5_dir, h5_file)
        for h5_file in h5_files
        if h5_file.split("-01Z")[0] in test_case_ids
    ]
    validate_h5_files = [
        os.path.join(h5_dir, h5_file)
        for h5_file in h5_files
        if h5_file.split("-01Z")[0] in validate_case_ids
    ]

    train_dataset = PatientDataset(
        train_case_ids,
        train_h5_files,
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
    )

    test_dataset = PatientDataset(
        test_case_ids,
        test_h5_files,
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )

    validate_dataset = PatientDataset(
        validate_case_ids,
        validate_h5_files,
        X_validate.reset_index(drop=True),
        y_validate.reset_index(drop=True),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
    )
    validate_loader = DataLoader(
        validate_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
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
