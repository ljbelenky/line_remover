#!/usr/bin/env python3
"""
Generate input patches by blending target patches with real ruled line examples.
Uses multiply blend to add realistic ruled lines to clean sketch patches.
"""

import os
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PATCHES_DIR = os.path.join(DATA_DIR, 'patches')
TRAIN_TARGET_DIR = os.path.join(PATCHES_DIR, 'train', 'target')
TRAIN_INPUT_DIR = os.path.join(PATCHES_DIR, 'train', 'input')
VAL_TARGET_DIR = os.path.join(PATCHES_DIR, 'validation', 'target')
VAL_INPUT_DIR = os.path.join(PATCHES_DIR, 'validation', 'input')
EXAMPLE_LINES_DIR = os.path.join(PATCHES_DIR, 'example_lines')


def get_example_lines_list():
    """Get list of all example line patches."""
    return [f for f in os.listdir(EXAMPLE_LINES_DIR)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]


def multiply_blend(target_img, lines_img):
    """Multiply blend target with squared lines.

    Squaring the lines makes them darker/more prominent before blending,
    better matching the appearance of real ruled paper.
    """
    target_arr = np.array(target_img).astype(np.float32) / 255.0
    lines_arr = np.array(lines_img).astype(np.float32) / 255.0

    blended = target_arr * (lines_arr ** 2)

    # Apply smoothstep contrast curve: darks darker, lights brighter, 0/0.5/1 fixed
    blended = 3 * blended**2 - 2 * blended**3

    blended = (blended * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def process_single_target(args):
    """Process a single target image, creating multiple blended inputs."""
    target_path, output_dir, example_files, repeat = args

    try:
        target_img = Image.open(target_path).convert('RGB')
        base_name = os.path.splitext(os.path.basename(target_path))[0]

        for i in range(repeat):
            # Select random example lines patch
            example_file = np.random.choice(example_files)
            example_path = os.path.join(EXAMPLE_LINES_DIR, example_file)
            lines_img = Image.open(example_path).convert('RGB')

            # Random horizontal flip (50% chance)
            if np.random.random() < 0.5:
                lines_img = lines_img.transpose(Image.FLIP_LEFT_RIGHT)

            # Random vertical flip (50% chance)
            if np.random.random() < 0.5:
                lines_img = lines_img.transpose(Image.FLIP_TOP_BOTTOM)

            # Multiply blend
            blended = multiply_blend(target_img, lines_img)

            # Save with suffix
            output_name = f"{base_name}_{i:02d}.jpg"
            output_path = os.path.join(output_dir, output_name)
            blended.save(output_path, quality=95)

        return repeat
    except Exception as e:
        print(f"Error processing {target_path}: {e}")
        return 0


def generate_inputs(target_dir, output_dir, repeat=1, limit=None, workers=4):
    """Generate input patches from target patches using example lines."""

    os.makedirs(output_dir, exist_ok=True)

    # Get list of example line patches
    example_files = get_example_lines_list()
    print(f"Found {len(example_files)} example line patches")

    if len(example_files) == 0:
        print("Error: No example line patches found!")
        return

    # Get target patches
    target_files = sorted([f for f in os.listdir(target_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if limit and limit < len(target_files):
        target_files = target_files[:limit]

    print(f"Processing {len(target_files)} target patches")
    print(f"Creating {repeat} input(s) per target")
    print(f"Total inputs to generate: {len(target_files) * repeat}")

    # Build task list
    tasks = []
    for filename in target_files:
        target_path = os.path.join(target_dir, filename)
        tasks.append((target_path, output_dir, example_files, repeat))

    # Process with parallel workers
    total_generated = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_target, task): task for task in tasks}

        for i, future in enumerate(as_completed(futures)):
            count = future.result()
            total_generated += count
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(tasks)} targets, {total_generated} inputs generated")

    print(f"\nDone! Generated {total_generated} input patches in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate input patches from targets using real line examples')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of inputs to generate per target (default: 1)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of target images to process')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--validation', action='store_true',
                        help='Process validation set instead of training set')
    args = parser.parse_args()

    if args.validation:
        target_dir = VAL_TARGET_DIR
        output_dir = VAL_INPUT_DIR
        print("Processing validation set...")
    else:
        target_dir = TRAIN_TARGET_DIR
        output_dir = TRAIN_INPUT_DIR
        print("Processing training set...")

    generate_inputs(
        target_dir=target_dir,
        output_dir=output_dir,
        repeat=args.repeat,
        limit=args.limit,
        workers=args.workers
    )


if __name__ == '__main__':
    main()
