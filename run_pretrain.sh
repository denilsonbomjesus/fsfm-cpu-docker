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
PRETRAIN_DATA_PATH="../../../datasets/pretrain_datasets/${DATASET_ID}"
# O diretório de saída será relativo ao diretório onde o script python é executado.
# Como o script python é executado em src/fsfm-3c/pretrain, o caminho relativo é ajustado.
OUTPUT_DIR="../pretrain/output_${DATASET_ID}"

# Source o arquivo de configuração para outros parâmetros como BATCH_SIZE, EPOCHS, etc.
# PRETRAIN_DATA_PATH e OUTPUT_DIR serão sobrescritos por este script.
source /app/config_pretraining.sh

# Ensure output directory exists and set correct ownership
mkdir -p "${OUTPUT_DIR}"
chown -R 1000:1000 "${OUTPUT_DIR}"

# Change to the pre-training script directory
cd /app/src/fsfm-3c/pretrain

# Execute o script de pré-treinamento com parâmetros
python main_pretrain.py \
  --batch_size "${BATCH_SIZE}" \
  --accum_iter "${ACCUM_ITER}" \
  --epochs "${EPOCHS}" \
  --model "${MODEL_NAME}" \
  --mask_ratio "${MASK_RATIO}" \
  --pretrain_data_path "${PRETRAIN_DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_workers "${NUM_WORKERS}"
