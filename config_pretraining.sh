#!/bin/bash

# Configuration for pre-training the FSFM model on CPU

# Batch size: Essential to avoid running out of RAM
export BATCH_SIZE=4

# Accumulate gradient iterations: For increasing effective batch size under memory constraints
export ACCUM_ITER=1

# Number of epochs: Sufficient to show that it "ran"
export EPOCHS=5

# Model name
export MODEL_NAME="fsfm_vit_base_patch16"

# Mask ratio
export MASK_RATIO=0.75



# Number of data loading workers: Essential for avoiding multiprocessing errors in emulation
export NUM_WORKERS=0

# Command to execute main_pretrain.py with these parameters
# Example usage: bash -c "source /app/config_pretraining.sh && cd /app/src/fsfm-3c/pretrain && python main_pretrain.py \
#   --batch_size ${BATCH_SIZE} \
#   --accum_iter ${ACCUM_ITER} \
#   --epochs ${EPOCHS} \
#   --model ${MODEL_NAME} \
#   --mask_ratio ${MASK_RATIO} \
#   --pretrain_data_path ${PRETRAIN_DATA_PATH} \
#   --output_dir ${OUTPUT_DIR} \
#   --num_workers ${NUM_WORKERS}"
