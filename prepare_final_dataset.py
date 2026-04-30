#!/usr/bin/env python3
"""
Prepare training and validation dataset from successfully processed images.

For each image in results/compare (curated good results):
- Input: the original ruled image from Sketches/ruled_rgb/
- Target: the model output from results/final/
- Extract random 512x512 patches with random flips
- Split 80/20 at the image level into data/train and data/validation
"""

import os
import sys
import argparse
import random
from multiprocessing import Pool
from PIL import Image

PATCH_SIZE = 512

COMPARE_DIR = 'results/compare'
INPUT_DIR = 'Sketches/ruled_rgb'
TARGET_DIR = 'results/final'

TRAIN_INPUT_DIR = 'data/patches/train/input'
TRAIN_TARGET_DIR = 'data/patches/train/target'
VAL_INPUT_DIR = 'data/patches/validation/input'
VAL_TARGET_DIR = 'data/patches/validation/target'


def process_image(args):
    """Process a single image: extract patches and save them."""
    fname, num_patches, seed, input_dir, target_dir = args
    rng = random.Random(seed)

    input_path = os.path.join(INPUT_DIR, fname)
    target_path = os.path.join(TARGET_DIR, fname)

    input_img = Image.open(input_path).convert('RGB')
    target_img = Image.open(target_path).convert('RGB')

    # Ensure same size
    if input_img.size != target_img.size:
        target_img = target_img.resize(input_img.size, Image.Resampling.LANCZOS)

    # Skip images too small for patches
    w, h = input_img.size
    if w < PATCH_SIZE or h < PATCH_SIZE:
        return fname, 0, f"Skipping {fname} ({w}x{h}) — too small"

    base = os.path.splitext(fname)[0]
    count = 0
    min_dist = PATCH_SIZE // 2
    positions = []

    for j in range(num_patches):
        # Try to find a position at least PATCH_SIZE/2 away from all others
        for attempt in range(100):
            x = rng.randint(0, w - PATCH_SIZE)
            y = rng.randint(0, h - PATCH_SIZE)
            if all(abs(x - px) >= min_dist or abs(y - py) >= min_dist for px, py in positions):
                break
        positions.append((x, y))

        input_patch = input_img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
        target_patch = target_img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))

        if rng.random() < 0.5:
            input_patch = input_patch.transpose(Image.FLIP_LEFT_RIGHT)
            target_patch = target_patch.transpose(Image.FLIP_LEFT_RIGHT)

        if rng.random() < 0.5:
            input_patch = input_patch.transpose(Image.FLIP_TOP_BOTTOM)
            target_patch = target_patch.transpose(Image.FLIP_TOP_BOTTOM)

        patch_name = f"{base}_p{j:03d}.jpg"
        input_patch.save(os.path.join(input_dir, patch_name), quality=95)
        target_patch.save(os.path.join(target_dir, patch_name), quality=95)
        count += 1

    return fname, count, None


def main():
    parser = argparse.ArgumentParser(description='Prepare final training dataset')
    parser.add_argument('--repeat', type=int, default=10,
                        help='Number of patches per image (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--workers', type=int, default=12,
                        help='Number of parallel workers (default: 12)')
    args = parser.parse_args()

    random.seed(args.seed)

    # Start from compare directory (curated list) and find matches in input/target
    compare_files = sorted(os.listdir(COMPARE_DIR))
    input_files = set(os.listdir(INPUT_DIR))
    target_files = set(os.listdir(TARGET_DIR))

    valid_files = []
    skipped = []
    for fname in compare_files:
        if fname in input_files and fname in target_files:
            valid_files.append(fname)
        else:
            skipped.append(fname)

    if skipped:
        print(f"WARNING: {len(skipped)} files in {COMPARE_DIR} missing from input/target dirs:")
        for s in skipped:
            missing = []
            if s not in input_files:
                missing.append(INPUT_DIR)
            if s not in target_files:
                missing.append(TARGET_DIR)
            print(f"  {s} — not in {', '.join(missing)}")

    if not valid_files:
        print("No matching images found.")
        sys.exit(1)

    print(f"Using {len(valid_files)} images from {COMPARE_DIR} (of {len(compare_files)} total)")
    print(f"Patches per image: {args.repeat}")

    # Split at image level: 80% train, 20% validation
    random.shuffle(valid_files)
    split_idx = int(len(valid_files) * 0.8)
    train_files = sorted(valid_files[:split_idx])
    val_files = sorted(valid_files[split_idx:])

    print(f"Train images: {len(train_files)}, Validation images: {len(val_files)}")

    # Create output directories
    for d in [TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, VAL_INPUT_DIR, VAL_TARGET_DIR]:
        os.makedirs(d, exist_ok=True)

    # Clear existing patches
    for d in [TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, VAL_INPUT_DIR, VAL_TARGET_DIR]:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    # Build work items with per-image seeds for reproducibility
    work_items = []
    for fname in train_files:
        seed = random.randint(0, 2**31)
        work_items.append((fname, args.repeat, seed, TRAIN_INPUT_DIR, TRAIN_TARGET_DIR))
    for fname in val_files:
        seed = random.randint(0, 2**31)
        work_items.append((fname, args.repeat, seed, VAL_INPUT_DIR, VAL_TARGET_DIR))

    print(f"Processing with {args.workers} workers...")

    total_train = 0
    total_val = 0

    with Pool(args.workers) as pool:
        for fname, count, warning in pool.imap_unordered(process_image, work_items):
            if warning:
                print(f"  {warning}")
            elif count > 0:
                split = 'train' if fname in set(train_files) else 'validation'
                if split == 'train':
                    total_train += count
                else:
                    total_val += count
                print(f"  [{split}] {fname}: {count} patches")

    print(f"\nDone! Train: {total_train} patches, Validation: {total_val} patches")


if __name__ == '__main__':
    main()
