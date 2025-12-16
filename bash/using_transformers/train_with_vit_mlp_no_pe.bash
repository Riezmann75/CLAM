#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ngpus=1:mem=32GB
#PBS -q normal
#PBS -P 12004268
#PBS -N finetuning_vit_no_pe
#PBS -j oe
#PBS -o /home/users/ntu/giabaoca/myProject-12004268/giabao/CLAM/log/transformers/train_with_vit_mlp_no_pe.txt
#PBS -M giabao.cao@ntu.edu.sg
#PBS -m abe

# 1. Navigate to the directory where the job was submitted
cd $PBS_O_WORKDIR

#activating conda environment
source activate base
conda activate clam_latest

# 4. Set Environment Variables
export HF_HOME=.hf_cache

# 5. Run the script
# Using backslashes (\) to break the long command for readability
python run.py \
    --batch_size 16 \
    --num_transformer_layers 2 \
    --config_path yaml/transformers/model_config_no_pe.yaml \
    --hidden_dim 1024 \
    --feature_dir wsi_patches/BLCA/features/BLCA_vit_mlp \
    --h5_dir wsi_patches/BLCA/patches/ \
    --clean_csv_path dataset_csv/tcga_blca_all_clean.csv \
    --encoder vit_mlp \
    --log_path experiments/vit_finetuning/result_vit_mlp_no_pe.jsonl \
    --exp_desc "Fine tune last MLP ViT without using PE: hidden dim 1024, default num transformer"
