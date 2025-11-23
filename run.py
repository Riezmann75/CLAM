import numpy as np
import torch

from lib.grid_search import GridSearch, SearchSpace
from lib.grid_search import SearchSpace
from lib.models import (
    NLL,
    GenomicEncoder,
    ImageEncoder,
    ImageEncoder,
    ResnetEncoder,
    SurvivalModel,
)
from lib.pre_process import load_dataset
from lib.train import train_model_with_config
from lib.utils import decorate_optimizer

import argparse

parser = argparse.ArgumentParser(
    description="Run the survival model training with grid search."
)
parser.add_argument(
    "--batch_size", type=int, default=8, help="Batch size for data loading"
)
parser.add_argument(
    "--hidden_dim", type=int, default=128, help="Hidden dimension size for encoders"
)
parser.add_argument(
    "--feature_dir",
    type=str,
    default="wsi_patches/BLCA/features/BLCA_resnet50",
    help="Directory for feature files",
)
parser.add_argument(
    "--h5_dir",
    type=str,
    default="wsi_patches/BLCA/patches/",
    help="Directory for h5 patch files",
)
parser.add_argument(
    "--clean_csv_path",
    type=str,
    default="dataset_csv/tcga_blca_all_clean.csv",
    help="Path to the cleaned CSV file",
)
parser.add_argument(
    "--encoder",
    choices=["resnet", "vit", "plip"],
    default="resnet",
    help="Type of path encoder to use",
)
parser.add_argument(
    "--log_path",
    type=str,
    default="experiments/result_logs.jsonl",
    help="Path to save the training logs",
)

args = parser.parse_args()

batch_size = args.batch_size
hidden_dim = args.hidden_dim
features_dir = args.feature_dir
h5_dir = args.h5_dir
clean_csv_path = args.clean_csv_path
encoder_type = args.encoder

if "resnet" in encoder_type:
    assert "resnet" in features_dir, "Feature directory does not match encoder type"
elif "vit" in encoder_type:
    assert "vit" in features_dir, "Feature directory does not match encoder type"
else:
    assert "plip" in features_dir, "Feature directory does not match encoder type"

processed_data = load_dataset(
    clean_csv_path=clean_csv_path,
    h5_dir=h5_dir,
    feature_dir=features_dir,
    batch_size=batch_size,
)

if args.encoder == "resnet":
    path_enc = ResnetEncoder(hidden_dim=hidden_dim)
elif args.encoder == "vit":
    path_enc = ImageEncoder(hidden_dim=hidden_dim)
elif args.encoder == "plip":
    path_enc = ImageEncoder(hidden_dim=hidden_dim)
else:
    raise ValueError(f"Unknown encoder type: {args.encoder}")

geno_enc = GenomicEncoder(
    df=processed_data["filtered_df"],
    categorical_cols=processed_data["categorical_cols"],
    numeric_cols=processed_data["numeric_cols"],
    hidden_dim=hidden_dim,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss = NLL()

search_space = SearchSpace.model_validate(
    {
        "learning_rates": np.arange(6e-5, 3e-4, step=2e-5).tolist(),
        "weight_decays": [1e-4],
        "optimizers": [
            decorate_optimizer(torch.optim.Adam),
        ],
        "num_epochs": [50],
    }
)

grid_searcher = GridSearch(search_space, device=device)
grid_searcher(
    Model=SurvivalModel,
    model_init_args={
        "path_encoder": path_enc.to(device),
        "geno_encoder": geno_enc.to(device),
        "hidden_dim": hidden_dim,
    },
    train_fn=train_model_with_config,
    loss_fn=NLL(),
    train_loader=processed_data["train_loader"],
    validation_loader=processed_data["validate_loader"],
    test_loader=processed_data["test_loader"],
)

# plot_top_configs(experiment_path=os.path.join(os.getcwd(), "experiments"))
