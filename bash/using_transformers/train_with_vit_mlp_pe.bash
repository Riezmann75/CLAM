#!/bin/bash
#PBS -l walltime=4:00:00
#PBS -l select=1:ngpus=1:mem=32GB
#PBS -q normal
#PBS -P 12004268
#PBS -N finetuning_vit_mlp_pe_4_1024
#PBS -j oe
#PBS -o /home/users/ntu/giabaoca/myProject-12004268/giabao/CLAM/log/transformers/train_with_vit_mlp_pe_4_1024.txt
#PBS -M giabao.cao@ntu.edu.sg
#PBS -m abe

# 1. Change to the project directory
cd /home/users/ntu/giabaoca/myProject-12004268/giabao/CLAM

# 2. Activate Conda Environment
# It is often safer to source the conda.sh script directly if 'source activate' fails,
# but if this works for your cluster, keep it.
source activate base
conda activate clam_latest

# 3. Debugging: Check loaded environment
# conda info --envs

# 4. Set Environment Variables
export HF_HOME=.hf_cache

# 5. Run the Training Script
python run.py \
    --batch_size 16 \
    --num_transformer_layers 1 \
    --config_path yaml/transformers/model_config_pe.yaml \
    --hidden_dim 512 \
    --num_workers 4 \
    --feature_dir wsi_patches/BLCA/features/BLCA_vit_mlp_20x \
    --h5_dir wsi_patches/BLCA/patches/ \
    --clean_csv_path dataset_csv/tcga_blca_all_clean.csv \
    --encoder vit_mlp \
    --log_path experiments/vit_finetuning/result_vit_mlp_pe_new_pe.jsonl \
    --exp_desc "Fine tune last MLP ViT with PE, hidden dim 512, 1 transformer encoders" \
    --is_model_saved true \
    --model_name finetune_last_mlp.pth
