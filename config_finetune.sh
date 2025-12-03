#!/bin/bash

# Configuration for fine-tuning the FSFM model on CPU

# Batch size
export FT_BATCH_SIZE=4

# Number of epochs
export FT_EPOCHS=5

# Path to the pre-trained checkpoint
export FT_FINETUNE_CHECKPOINT="/app/src/fsfm-3c/pretrain/output_cpu_test/checkpoint-4.pth"



# Device to use (cpu)
export FT_DEVICE="cpu"

# Number of data loading workers (0 for CPU emulation)
export FT_NUM_WORKERS=0
