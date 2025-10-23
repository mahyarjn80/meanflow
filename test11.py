#!/usr/bin/env python3
"""
Script for computing FID statistics only.

This script:
1. Loads ImageNet data from the specified folder
2. Computes FID statistics and saves them

Usage:
    python test11.py --imagenet_root /path/to/imagenet --output_dir /path/to/output --image_size 256
"""

import logging
import os

import jax
from absl import app, flags

# Initialize JAX distributed processing
jax.distributed.initialize()

from utils.fid_util import compute_fid_stats
from utils.logging_util import log_for_0

FLAGS = flags.FLAGS
flags.DEFINE_string('imagenet_root', '/path/to/imagenet', 'Path to ImageNet dataset root')
flags.DEFINE_string('output_dir', '/path/to/output', 'Output directory for FID stats')
flags.DEFINE_integer('image_size', 256, 'Image size for processing (e.g., 256, 512, 1024)')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite existing files')


def main(argv):
    """Main function."""
    del argv  # Unused
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Validate paths
    if not os.path.exists(FLAGS.imagenet_root):
        raise ValueError(f"ImageNet root path does not exist: {FLAGS.imagenet_root}")
    

    log_for_0(f"Output directory: {FLAGS.output_dir}")
    
    # Log JAX setup
    local_device_count = jax.local_device_count()
    log_for_0(f"JAX distributed setup: process {jax.process_index()}/{jax.process_count()}, "
             f"local devices: {local_device_count}, total devices: {jax.device_count()}")

    # Compute FID statistics only
    log_for_0("="*50)
    log_for_0("COMPUTING FID STATISTICS")
    log_for_0("="*50)

    fid_stats_path = compute_fid_stats(
        imagenet_root=FLAGS.imagenet_root,
        output_dir=FLAGS.output_dir,
        image_size=FLAGS.image_size,
        overwrite=FLAGS.overwrite
    )

    log_for_0(f"FID statistics computed and saved to: {fid_stats_path}")
    log_for_0("="*50)
    log_for_0("COMPUTATION COMPLETED SUCCESSFULLY")
    log_for_0("="*50)


if __name__ == '__main__':
    app.run(main) 