#!/bin/bash
#SBATCH --job-name=fsfm_finetune_fas
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Set the root directory for the dataset
DATA_PATH="./lfw_mock"

# Set the output directory for logs and models
OUTPUT_DIR="./src/fsfm-3c/finuetune/cross_domain_FAS/output_finetune_cpu_test_fas/"

# Create the output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Execute the fine-tuning script
python3 ./src/fsfm-3c/finuetune/cross_domain_FAS/train_vit.py \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --epochs 1 \
    --batch_size 4 \
    --model vit_small_patch16 \
    --device cpu \
    --num_workers 0
