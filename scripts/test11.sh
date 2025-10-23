#!/bin/bash

# Configuration for data preparation (can be overridden via env before calling this script)
export IMAGENET_ROOT="${IMAGENET_ROOT:-/orcd/datasets/legacy/001/imagenet/images_complete/ilsvrc}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/mahyarjn/meanflow/output}"
export LOG_DIR="${LOG_DIR:-/home/mahyarjn/meanflow/logs}"

# Validate required paths
if [ ! -d "$IMAGENET_ROOT" ]; then
    echo "ERROR: IMAGENET_ROOT does not exist or is not a directory: $IMAGENET_ROOT"
    echo "Please set IMAGENET_ROOT to the folder that contains 'train' and 'val' subfolders."
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR" || { echo "ERROR: Failed to create OUTPUT_DIR: $OUTPUT_DIR"; exit 1; }
fi

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR" || { echo "ERROR: Failed to create LOG_DIR: $LOG_DIR"; exit 1; }
fi

export BATCH_SIZE=128
export VAE_TYPE="mse"

export now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
export JOBNAME=prepare_data_${now}_${salt}_$1
export LOG_DIR=$LOG_DIR/$USER/$JOBNAME

mkdir -p "${LOG_DIR}"
chmod 777 -R "${LOG_DIR}"

# Image size configuration (common sizes: 256, 512, 1024)
# Corresponding latent sizes will be: 32x32, 64x64, 128x128
IMAGE_SIZE=${IMAGE_SIZE:-256}  # Can be overridden via environment variable

# Computation flags (can be overridden via environment variables)
COMPUTE_LATENT=${COMPUTE_LATENT:-True}  # Whether to compute latent dataset
COMPUTE_FID=${COMPUTE_FID:-False}       # Whether to compute FID statistics

# Calculate latent size for display
LATENT_SIZE=$((IMAGE_SIZE / 8))

echo "=============================================="
echo "Data Preparation Configuration"
echo "=============================================="
echo "ImageNet Root: $IMAGENET_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "VAE Type: $VAE_TYPE"
echo "Image Size: $IMAGE_SIZE -> Latent Size: ${LATENT_SIZE}x${LATENT_SIZE}"
echo "Compute Latent: $COMPUTE_LATENT"
echo "Compute FID: $COMPUTE_FID"
if [ "$COMPUTE_FID" = "True" ]; then
    echo "FID: Using ALL training samples"
fi
echo "=============================================="

python3 test11.py \
    --imagenet_root="$IMAGENET_ROOT" \
    --output_dir="$OUTPUT_DIR" \
    --image_size="$IMAGE_SIZE" \
    2>&1 | tee -a "$LOG_DIR/output.log"

echo "=============================================="
echo "Data preparation completed!"
echo "Check logs at: $LOG_DIR/output.log"
if [ "$COMPUTE_LATENT" = "True" ]; then
    echo "Latent dataset saved to: $OUTPUT_DIR"
fi
if [ "$COMPUTE_FID" = "True" ]; then
    echo "FID stats saved to: $OUTPUT_DIR/imagenet_${IMAGE_SIZE}_fid_stats.npz"
fi
echo "==============================================" 
