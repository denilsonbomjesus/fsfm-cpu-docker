#!/bin/bash
#SBATCH --job-name=fsfm_finetune_fas
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Aceita o ID do dataset como primeiro argumento
DATASET_ID=$1

# Verifica se um ID de dataset foi fornecido
if [ -z "$DATASET_ID" ]; then
  echo "Uso: $0 <DATASET_ID>"
  exit 1
fi

# Set the root directory for the dataset dynamically
DATA_PATH="$(realpath ./datasets/finetune/FAS/${DATASET_ID})"

# Set the output directory for logs and models dynamically
OUTPUT_DIR="./src/fsfm-3c/finuetune/cross_domain_FAS/output_finetune_${DATASET_ID}/"

# Create the output directory if it doesn't exist and set correct ownership
mkdir -p "${OUTPUT_DIR}"
chown -R 1000:1000 "${OUTPUT_DIR}"

# Execute the fine-tuning script
python3 ./src/fsfm-3c/finuetune/cross_domain_FAS/train_vit.py \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 1 \
    --batch_size 4 \
    --model vit_small_patch16 \
    --device cpu \
    --num_workers 0 > "${OUTPUT_DIR}/log_detail.txt" 2>&1
