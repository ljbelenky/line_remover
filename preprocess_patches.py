#!/usr/bin/env python3
"""
Preprocess training images into patches for faster training.
Extracts random patches from lined/original image pairs.
"""

import os
import re
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Patches directories (new structure)
PATCHES_DIR = os.path.join(DATA_DIR, 'patches')
TRAIN_PATCHES_DIR = os.path.join(PATCHES_DIR, 'train')
VAL_PATCHES_DIR = os.path.join(PATCHES_DIR, 'validation')
INFERENCE_DIR = os.path.join(PATCHES_DIR, 'inference')
RULED_PATCHES_DIR = os.path.join(INFERENCE_DIR, 'ruled_input')
EXAMPLE_LINES_DIR = os.path.join(PATCHES_DIR, 'example_lines')

# Legacy paths (for source images)
SKETCHES_DIR = os.path.join(DATA_DIR, 'Sketches')
LINES_ADDED_DIR = os.path.join(SKETCHES_DIR, 'lines_added')
UNRULED_DIR = os.path.join(SKETCHES_DIR, 'Unruled')
VALIDATION_DIR = os.path.join(SKETCHES_DIR, 'validation')
RULED_DIR = os.path.join(SKETCHES_DIR, 'ruled_rgb')

# Patch parameters
PATCH_SIZE = 512
MIN_IMAGE_SIZE = 512


def get_base_name(filename):
    """Extract base name from a filename by removing the suffix like _00001."""
    name_without_ext = os.path.splitext(filename)[0]
    match = re.match(r'(.+)_\d{5}$', name_without_ext)
    if match:
        return match.group(1)
    return name_without_ext


def find_original_image(lined_filename):
    """Find the original unruled image for a given lined image."""
    base_name = get_base_name(lined_filename)

    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        original_path = os.path.join(UNRULED_DIR, base_name + ext)
        if os.path.exists(original_path):
            return original_path
    return None


def extract_patches_from_pair(lined_path, original_path, num_patches, output_dir):
    """Extract random patches from a lined/original image pair."""
    try:
        lined_img = Image.open(lined_path).convert('RGB')
        original_img = Image.open(original_path).convert('RGB')

        width, height = lined_img.size

        # Skip if image is too small
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            return 0

        # Resize original if needed
        if original_img.size != (width, height):
            original_img = original_img.resize((width, height), Image.Resampling.LANCZOS)

        base_name = os.path.splitext(os.path.basename(lined_path))[0]
        patches_saved = 0

        for i in range(num_patches):
            # Random crop location
            max_x = width - PATCH_SIZE
            max_y = height - PATCH_SIZE

            if max_x < 0 or max_y < 0:
                continue

            x = np.random.randint(0, max_x + 1)
            y = np.random.randint(0, max_y + 1)

            # Extract patches
            lined_patch = lined_img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            original_patch = original_img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))

            # Save patches
            patch_name = f"{base_name}_p{i:03d}"
            lined_patch.save(os.path.join(output_dir, 'input', f"{patch_name}.jpg"), quality=95)
            original_patch.save(os.path.join(output_dir, 'target', f"{patch_name}.jpg"), quality=95)
            patches_saved += 1

        return patches_saved
    except Exception as e:
        print(f"Error processing {lined_path}: {e}")
        return 0


def process_image_pair(args):
    """Wrapper for parallel processing."""
    lined_path, original_path, num_patches, output_dir = args
    return extract_patches_from_pair(lined_path, original_path, num_patches, output_dir)


def extract_patches_from_single(image_path, num_patches, output_dir):
    """Extract random patches from a single image (no target pair)."""
    try:
        img = Image.open(image_path).convert('RGB')
        width, height = img.size

        # Skip if image is too small
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            return 0

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        patches_saved = 0

        for i in range(num_patches):
            # Random crop location
            max_x = width - PATCH_SIZE
            max_y = height - PATCH_SIZE

            if max_x < 0 or max_y < 0:
                continue

            x = np.random.randint(0, max_x + 1)
            y = np.random.randint(0, max_y + 1)

            # Extract patch
            patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))

            # Save patch
            patch_name = f"{base_name}_p{i:03d}.jpg"
            patch.save(os.path.join(output_dir, patch_name), quality=95)
            patches_saved += 1

        return patches_saved
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0


def process_single_image(args):
    """Wrapper for parallel processing of single images."""
    image_path, num_patches, output_dir = args
    return extract_patches_from_single(image_path, num_patches, output_dir)


def main(num_patches=16, workers=4):
    """Preprocess all training and validation images into patches."""

    # Create output directories
    for subdir in [TRAIN_PATCHES_DIR, VAL_PATCHES_DIR]:
        os.makedirs(os.path.join(subdir, 'input'), exist_ok=True)
        os.makedirs(os.path.join(subdir, 'target'), exist_ok=True)

    # Process training images
    print(f"Processing training images from {LINES_ADDED_DIR}...")
    train_files = [f for f in os.listdir(LINES_ADDED_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    train_tasks = []
    for filename in train_files:
        lined_path = os.path.join(LINES_ADDED_DIR, filename)
        original_path = find_original_image(filename)
        if original_path:
            train_tasks.append((lined_path, original_path, num_patches, TRAIN_PATCHES_DIR))

    print(f"Found {len(train_tasks)} training image pairs")

    # Process validation images
    print(f"Processing validation images from {VALIDATION_DIR}...")
    val_files = [f for f in os.listdir(VALIDATION_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    val_tasks = []
    for filename in val_files:
        lined_path = os.path.join(VALIDATION_DIR, filename)
        original_path = find_original_image(filename)
        if original_path:
            val_tasks.append((lined_path, original_path, num_patches, VAL_PATCHES_DIR))

    print(f"Found {len(val_tasks)} validation image pairs")

    # Process all tasks
    all_tasks = train_tasks + val_tasks
    total_patches = 0

    print(f"\nExtracting {num_patches} patches per image with {workers} workers...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_image_pair, task): task for task in all_tasks}

        for i, future in enumerate(as_completed(futures)):
            patches = future.result()
            total_patches += patches
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(all_tasks)} images, {total_patches} patches saved")

    print(f"\nDone! Saved {total_patches} total patches")
    print(f"Training patches: {TRAIN_PATCHES_DIR}")
    print(f"Validation patches: {VAL_PATCHES_DIR}")


def process_ruled(num_patches=10, workers=4):
    """Extract patches from ruled images (no target pairs)."""
    os.makedirs(RULED_PATCHES_DIR, exist_ok=True)

    print(f"Processing ruled images from {RULED_DIR}...")
    ruled_files = [f for f in os.listdir(RULED_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    tasks = []
    for filename in ruled_files:
        image_path = os.path.join(RULED_DIR, filename)
        tasks.append((image_path, num_patches, RULED_PATCHES_DIR))

    print(f"Found {len(tasks)} ruled images")
    print(f"Extracting {num_patches} patches per image with {workers} workers...")

    total_patches = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_image, task): task for task in tasks}

        for i, future in enumerate(as_completed(futures)):
            patches = future.result()
            total_patches += patches
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(tasks)} images, {total_patches} patches saved")

    print(f"\nDone! Saved {total_patches} ruled patches to {RULED_PATCHES_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess images into patches')
    parser.add_argument('--patches', type=int, default=16,
                        help='Number of patches per image (default: 16)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--ruled', action='store_true',
                        help='Process ruled images only (10 patches per image)')
    args = parser.parse_args()

    if args.ruled:
        process_ruled(num_patches=10, workers=args.workers)
    else:
        main(num_patches=args.patches, workers=args.workers)
