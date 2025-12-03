#!/bin/bash

# Aceita o ID do dataset como primeiro argumento
DATASET_ID=$1

# Verifica se um ID de dataset foi fornecido
if [ -z "$DATASET_ID" ]; then
  echo "Uso: $0 <DATASET_ID>"
  exit 1
fi

# Define os caminhos de dados e saída dinamicamente, com base no DATASET_ID
# O caminho para o dataset será relativo à raiz do projeto.
FT_DATA_PATH="../../../../datasets/finetune/DfD/${DATASET_ID}"
# O diretório de saída será relativo à raiz do projeto.
FT_OUTPUT_DIR="./src/fsfm-3c/finuetune/cross_dataset_DfD/output_finetune_${DATASET_ID}"

# Source the configuration file for other parameters
source /app/config_finetune.sh

echo "Starting FSFM Fine-Tuning..."
echo "Batch Size: ${FT_BATCH_SIZE}"
echo "Epochs: ${FT_EPOCHS}"
echo "Finetune Checkpoint: ${FT_FINETUNE_CHECKPOINT}"
echo "Data Path: ${FT_DATA_PATH}"
echo "Output Directory: ${FT_OUTPUT_DIR}"
echo "Device: ${FT_DEVICE}"
echo "Num Workers: ${FT_NUM_WORKERS}"

# Ensure output directory exists and set correct ownership
mkdir -p "${FT_OUTPUT_DIR}"
chown -R 1000:1000 "${FT_OUTPUT_DIR}"
python main_finetune_DfD.py \
  --batch_size "${FT_BATCH_SIZE}" \
  --epochs "${FT_EPOCHS}" \
  --finetune "${FT_FINETUNE_CHECKPOINT}" \
  --finetune_data_path "${FT_DATA_PATH}" \
  --output_dir "${FT_OUTPUT_DIR}" \
  --device "${FT_DEVICE}" \
  --num_workers "${FT_NUM_WORKERS}" > "${FT_OUTPUT_DIR}/log_detail.txt" 2>&1

echo "FSFM Fine-Tuning finished."

