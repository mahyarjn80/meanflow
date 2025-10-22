#!/bin/bash

# ImageNet Download Script Wrapper
# This script provides an easy way to download the ImageNet dataset

set -e

# Default values
OUTPUT_DIR="./imagenet"
USERNAME=""
ACCESS_KEY=""
SKIP_TRAIN=false
SKIP_VAL=false
SKIP_DEVKIT=false

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -o, --output-dir DIR     Output directory for ImageNet dataset (default: ./imagenet)"
    echo "  -u, --username USER      Username for ImageNet access"
    echo "  -k, --access-key KEY     Access key for ImageNet access"
    echo "  --skip-train             Skip downloading training data (~138GB)"
    echo "  --skip-val               Skip downloading validation data"
    echo "  --skip-devkit            Skip downloading development kit"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Note: You need to register at https://image-net.org/ to get access credentials."
    echo ""
    echo "Example:"
    echo "  $0 -o /data/imagenet -u myusername -k myaccesskey"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -u|--username)
            USERNAME="$2"
            shift 2
            ;;
        -k|--access-key)
            ACCESS_KEY="$2"
            shift 2
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-val)
            SKIP_VAL=true
            shift
            ;;
        --skip-devkit)
            SKIP_DEVKIT=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if Python script exists
if [ ! -f "download_imagenet.py" ]; then
    echo "Error: download_imagenet.py not found in current directory"
    exit 1
fi

# Build Python command
PYTHON_CMD="python3 download_imagenet.py --output_dir \"$OUTPUT_DIR\""

if [ -n "$USERNAME" ]; then
    PYTHON_CMD="$PYTHON_CMD --username \"$USERNAME\""
fi

if [ -n "$ACCESS_KEY" ]; then
    PYTHON_CMD="$PYTHON_CMD --access_key \"$ACCESS_KEY\""
fi

if [ "$SKIP_TRAIN" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --skip_train"
fi

if [ "$SKIP_VAL" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --skip_val"
fi

if [ "$SKIP_DEVKIT" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --skip_devkit"
fi

echo "=============================================="
echo "ImageNet Download Script"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Username: ${USERNAME:-'Not provided'}"
echo "Access key: ${ACCESS_KEY:+[PROVIDED]}"
echo "Skip train: $SKIP_TRAIN"
echo "Skip val: $SKIP_VAL"
echo "Skip devkit: $SKIP_DEVKIT"
echo "=============================================="

# Execute Python script
eval $PYTHON_CMD

echo "=============================================="
echo "Download completed!"
echo "ImageNet dataset saved to: $OUTPUT_DIR"
echo "=============================================="
