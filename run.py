import numpy as np
import torch

from lib.grid_search import GridSearch, SearchSpace
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
from lib.train_utils.utils import decorate_optimizer
import yaml

import argparse

torch.manual_seed(42)
np.random.seed(42)

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
    choices=["resnet", "vit", "vit_mlp", "plip", "simclr"],
    default="resnet",
    help="Type of path encoder to use",
)
parser.add_argument(
    "--log_path",
    type=str,
    default="experiments/result_logs.jsonl",
    help="Path to save the training logs",
)

parser.add_argument(
    "--config_path",
    type=str,
    default=None,
    help="Path to the model configuration YAML file",
)

args = parser.parse_args()

batch_size = args.batch_size
hidden_dim = args.hidden_dim
extracted_dir = args.feature_dir
h5_dir = args.h5_dir
clean_csv_path = args.clean_csv_path
encoder_type = args.encoder
log_path = args.log_path
config_path = args.config_path
if config_path is not None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {
        "architecture": [
            {"positional_encoding": True},
            {"gated_attention": False},
            {"transformer": True},
        ]
    }

hidden_dim = config.get("hidden_dim", hidden_dim)
use_positional_encoding = True
use_gated_attention = False
use_transformer = False
for item in config.get("architecture", []):
    if "positional_encoding" in item:
        use_positional_encoding = item["positional_encoding"]
    if "gated_attention" in item:
        use_gated_attention = item["gated_attention"]
    if "transformer" in item:
        use_transformer = item["transformer"]

if "resnet" in encoder_type:
    assert "resnet" in extracted_dir, "Feature directory does not match encoder type"
elif "vit" in encoder_type:
    assert "vit" in extracted_dir, "Feature directory does not match encoder type"
elif "vit_mlp" in encoder_type:
    assert "vit_mlp" in extracted_dir, "Feature directory does not match encoder type"
elif "plip" in encoder_type:
    assert "plip" in extracted_dir, "Feature directory does not match encoder type"
else:
    assert "simclr" in extracted_dir, "Feature directory does not match encoder type"

processed_data = load_dataset(
    clean_csv_path=clean_csv_path,
    extracted_dir=extracted_dir,
    batch_size=batch_size,
)

if args.encoder == "resnet":
    path_enc = ResnetEncoder(hidden_dim=hidden_dim)
elif args.encoder == "vit":
    path_enc = ImageEncoder(hidden_dim=hidden_dim)
elif args.encoder == "vit_mlp":
    path_enc = ImageEncoder(hidden_dim=hidden_dim)
elif args.encoder == "plip":
    path_enc = ImageEncoder(hidden_dim=hidden_dim)
elif args.encoder == "simclr":
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

grid_searcher = GridSearch(search_space, device=device, log_path=log_path)
grid_searcher(
    Model=SurvivalModel,
    model_init_args={
        "path_encoder": path_enc.to(device),
        "geno_encoder": geno_enc.to(device),
        "hidden_dim": hidden_dim,
        "use_positional_encoding": use_positional_encoding,
        "use_gated_attention": use_gated_attention,
        "use_transformer": use_transformer,
    },
    train_fn=train_model_with_config,
    loss_fn=NLL(),
    train_loader=processed_data["train_loader"],
    validation_loader=processed_data["validate_loader"],
    test_loader=processed_data["test_loader"],
)

# plot_top_configs(experiment_path=os.path.join(os.getcwd(), "experiments"))
