#!/bin/bash

# Source the configuration file
source /app/config_finetune.sh

echo "Starting FSFM Fine-Tuning..."
echo "Batch Size: ${FT_BATCH_SIZE}"
echo "Epochs: ${FT_EPOCHS}"
echo "Finetune Checkpoint: ${FT_FINETUNE_CHECKPOINT}"
echo "Data Path: ${FT_DATA_PATH}"
echo "Output Directory: ${FT_OUTPUT_DIR}"
echo "Device: ${FT_DEVICE}"
echo "Num Workers: ${FT_NUM_WORKERS}"

# Ensure output directory exists
mkdir -p "${FT_OUTPUT_DIR}"

# Navigate to the fine-tuning script directory and execute
cd /app/src/fsfm-3c/finuetune/cross_dataset_DfD/ && \
python main_finetune_DfD.py \
  --batch_size "${FT_BATCH_SIZE}" \
  --epochs "${FT_EPOCHS}" \
  --finetune "${FT_FINETUNE_CHECKPOINT}" \
  --finetune_data_path "${FT_DATA_PATH}" \
  --output_dir "${FT_OUTPUT_DIR}" \
  --device "${FT_DEVICE}" \
  --num_workers "${FT_NUM_WORKERS}"

echo "FSFM Fine-Tuning finished."
