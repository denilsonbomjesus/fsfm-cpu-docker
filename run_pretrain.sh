#!/bin/bash

# Source the configuration file
source /app/config_pretraining.sh

# Change to the pre-training script directory
cd /app/src/fsfm-3c/pretrain

# Execute the pre-training script with parameters from config
python main_pretrain.py \
  --batch_size "${BATCH_SIZE}" \
  --accum_iter "${ACCUM_ITER}" \
  --epochs "${EPOCHS}" \
  --model "${MODEL_NAME}" \
  --mask_ratio "${MASK_RATIO}" \
  --pretrain_data_path "${PRETRAIN_DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_workers "${NUM_WORKERS}"
