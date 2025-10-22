#!/usr/bin/env python3
"""
Script to download ImageNet dataset.

This script downloads the ImageNet ILSVRC2012 dataset which includes:
- Training images (1.2M images)
- Validation images (50K images)
- Development kit with metadata

Usage:
    python download_imagenet.py --output_dir /path/to/output --username YOUR_USERNAME --access_key YOUR_ACCESS_KEY

Note: You need to register at https://image-net.org/ to get access credentials.
"""

import argparse
import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from urllib.parse import urlparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ImageNet URLs (these may change, check https://image-net.org/download.php)
IMAGENET_URLS = {
    'train': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar',
    'val': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar',
    'devkit': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz'
}

def download_file(url, output_path, username=None, access_key=None, chunk_size=8192):
    """
    Download a file with progress tracking and resume capability.
    
    Args:
        url: URL to download from
        output_path: Local path to save the file
        username: Username for authentication
        access_key: Access key for authentication
        chunk_size: Size of chunks to download
    """
    headers = {}
    if username and access_key:
        headers['Authorization'] = f'Basic {username}:{access_key}'
    
    # Check if file already exists and get its size for resume
    resume_header = {}
    if os.path.exists(output_path):
        resume_header['Range'] = f'bytes={os.path.getsize(output_path)}-'
        logger.info(f"Resuming download of {os.path.basename(output_path)}")
    
    try:
        response = requests.get(url, headers={**headers, **resume_header}, stream=True)
        response.raise_for_status()
        
        # Handle partial content for resume
        mode = 'ab' if 'Range' in resume_header else 'wb'
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        
        with open(output_path, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rDownloading {os.path.basename(output_path)}: {progress:.1f}% ({downloaded_size:,}/{total_size:,} bytes)", end='', flush=True)
        
        print()  # New line after progress
        logger.info(f"Successfully downloaded {os.path.basename(output_path)}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        raise

def extract_tar_file(tar_path, extract_dir):
    """Extract a tar file to the specified directory."""
    logger.info(f"Extracting {os.path.basename(tar_path)}...")
    
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(extract_dir)
    
    logger.info(f"Successfully extracted {os.path.basename(tar_path)}")

def extract_tar_gz_file(tar_gz_path, extract_dir):
    """Extract a tar.gz file to the specified directory."""
    logger.info(f"Extracting {os.path.basename(tar_gz_path)}...")
    
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    
    logger.info(f"Successfully extracted {os.path.basename(tar_gz_path)}")

def validate_imagenet_structure(data_dir):
    """
    Validate that the ImageNet dataset has the expected structure.
    
    Args:
        data_dir: Path to the ImageNet dataset directory
        
    Returns:
        bool: True if structure is valid, False otherwise
    """
    expected_dirs = ['train', 'val']
    data_path = Path(data_dir)
    
    for dir_name in expected_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            logger.error(f"Missing directory: {dir_path}")
            return False
        
        # Check if train directory has subdirectories (class folders)
        if dir_name == 'train':
            subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
            if len(subdirs) < 1000:  # ImageNet has 1000 classes
                logger.warning(f"Train directory has only {len(subdirs)} subdirectories, expected ~1000")
        
        # Check if val directory has images
        if dir_name == 'val':
            val_images = list(dir_path.glob('*.JPEG'))
            if len(val_images) < 50000:  # ImageNet validation has 50K images
                logger.warning(f"Val directory has only {len(val_images)} images, expected ~50000")
    
    logger.info("ImageNet dataset structure validation passed")
    return True

def main():
    parser = argparse.ArgumentParser(description='Download ImageNet dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save the ImageNet dataset')
    parser.add_argument('--username', type=str,
                       help='Username for ImageNet access (register at https://image-net.org/)')
    parser.add_argument('--access_key', type=str,
                       help='Access key for ImageNet access')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip downloading training data (large file ~138GB)')
    parser.add_argument('--skip_val', action='store_true',
                       help='Skip downloading validation data')
    parser.add_argument('--skip_devkit', action='store_true',
                       help='Skip downloading development kit')
    parser.add_argument('--extract', action='store_true', default=True,
                       help='Extract downloaded archives (default: True)')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Validate dataset structure after download (default: True)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"ImageNet download starting...")
    logger.info(f"Output directory: {output_dir}")
    
    # Check if credentials are provided
    if not args.username or not args.access_key:
        logger.warning("No credentials provided. You may need to register at https://image-net.org/")
        logger.warning("Some downloads may fail without proper authentication.")
    
    # Download files
    downloaded_files = []
    
    if not args.skip_train:
        train_file = output_dir / 'ILSVRC2012_img_train.tar'
        logger.info("Downloading training data (this is a large file ~138GB)...")
        download_file(IMAGENET_URLS['train'], str(train_file), args.username, args.access_key)
        downloaded_files.append(train_file)
    
    if not args.skip_val:
        val_file = output_dir / 'ILSVRC2012_img_val.tar'
        logger.info("Downloading validation data...")
        download_file(IMAGENET_URLS['val'], str(val_file), args.username, args.access_key)
        downloaded_files.append(val_file)
    
    if not args.skip_devkit:
        devkit_file = output_dir / 'ILSVRC2012_devkit_t12.tar.gz'
        logger.info("Downloading development kit...")
        download_file(IMAGENET_URLS['devkit'], str(devkit_file), args.username, args.access_key)
        downloaded_files.append(devkit_file)
    
    # Extract files if requested
    if args.extract:
        logger.info("Extracting downloaded files...")
        
        for file_path in downloaded_files:
            if file_path.suffix == '.tar':
                extract_tar_file(str(file_path), str(output_dir))
            elif file_path.suffix == '.gz':
                extract_tar_gz_file(str(file_path), str(output_dir))
    
    # Validate dataset structure
    if args.validate and args.extract:
        logger.info("Validating dataset structure...")
        validate_imagenet_structure(output_dir)
    
    logger.info("ImageNet download completed!")
    logger.info(f"Dataset saved to: {output_dir}")
    
    if args.extract:
        logger.info("Expected structure:")
        logger.info("  imagenet/")
        logger.info("  ├── train/")
        logger.info("  │   ├── n01440764/")
        logger.info("  │   ├── n01443537/")
        logger.info("  │   └── ...")
        logger.info("  ├── val/")
        logger.info("  │   ├── ILSVRC2012_val_00000001.JPEG")
        logger.info("  │   ├── ILSVRC2012_val_00000002.JPEG")
        logger.info("  │   └── ...")
        logger.info("  └── ILSVRC2012_devkit_t12/")

if __name__ == '__main__':
    main()
