#!/usr/bin/env python3
"""
Add synthetic ruled lines to unruled sketch images.
"""

import os
import numpy as np
from PIL import Image
from ruler_generator import RulerGenerator

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Sketches')
UNRULED_DIR = os.path.join(DATA_DIR, 'Unruled')
OUTPUT_DIR = os.path.join(DATA_DIR, 'lines_added')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')


def count_existing_variations(output_dir, base_name):
    """Count how many variations already exist for a given base name."""
    if not os.path.exists(output_dir):
        return 0

    count = 0
    for filename in os.listdir(output_dir):
        name_without_ext = os.path.splitext(filename)[0]
        # Check if this file starts with our base name followed by _XXXXX
        if name_without_ext.startswith(base_name + '_'):
            try:
                suffix_str = name_without_ext.split('_')[-1]
                int(suffix_str)  # Validate it's a number
                count += 1
            except ValueError:
                continue
    return count


def get_next_suffix_for_image(output_dir, base_name):
    """Get the next available suffix number for a specific image base name."""
    existing_files = os.listdir(output_dir)
    max_suffix = 0

    for filename in existing_files:
        name_without_ext = os.path.splitext(filename)[0]
        # Check if this file starts with our base name followed by _
        if name_without_ext.startswith(base_name + '_'):
            try:
                suffix_str = name_without_ext.split('_')[-1]
                suffix_num = int(suffix_str)
                max_suffix = max(max_suffix, suffix_num)
            except ValueError:
                continue

    return max_suffix + 1


def calculate_line_count(height, target_spacing):
    """Calculate number of lines based on image height and target spacing.

    Args:
        height: Image height in pixels
        target_spacing: Target spacing between lines in pixels

    Returns:
        Number of lines to generate
    """
    return max(3, int(height / target_spacing))


def generate_random_ruler_params(shape, realistic=True):
    """Generate randomized parameters for ruler lines.

    Args:
        shape: (width, height) tuple
        realistic: If True, generate lines matching real ruled paper (light gray, straight).
                   If False, generate more varied synthetic lines.
    """
    width, height = shape

    if realistic:
        # Parameters matching real ruled paper scans
        # Real ruled paper typically has 200-300px spacing at high resolution
        # Use a range to add variety
        target_spacing = np.random.randint(180, 280)
        line_count = calculate_line_count(height, target_spacing)

        # Mix of cyan and gray lines like real notebooks
        line_color = np.random.choice(['cyan', 'gray', 'gray'])  # More gray
        return {
            'shape': shape,
            'line_width': np.random.choice([1, 2, 2, 3]),  # Include thicker lines
            'lines': line_count,  # Calculated based on spacing
            'v_offset': np.random.random(),
            'raggedness': np.random.uniform(-0.05, 0.05),  # Minimal raggedness
            'color': np.random.randint(200, 255),  # Strong visible lines
            'color_variation': np.random.randint(0, 15),  # Some variation
            'angle': np.random.uniform(-0.5, 0.5),  # Nearly horizontal
            'line_color': line_color,  # Mix of cyan and gray
            'add_margin': False,  # No margin line
            'waviness': 0.0,  # Perfectly straight
        }
    else:
        # More varied synthetic lines - also use spacing-based calculation
        target_spacing = np.random.randint(60, 200)
        line_count = calculate_line_count(height, target_spacing)

        return {
            'shape': shape,
            'line_width': np.clip(np.random.randint(-2, 4), 1, 4),
            'lines': line_count,  # Calculated based on spacing
            'v_offset': np.random.random(),
            'raggedness': -0.15 + np.random.random() / 2,
            'color': np.random.randint(60, 130),
            'color_variation': np.random.randint(-15, 16),
            'angle': np.random.randint(-7, 8),
            'line_color': None,  # Random color
            'add_margin': None,  # Random margin
            'waviness': np.random.uniform(0, 0.3),
        }


def blend_images(original, lines):
    """Blend the ruled lines with the original image using multiply blend.

    This works better than minimum blend for colored/cream paper because
    it multiplies the values rather than taking the minimum, ensuring
    lines are always visible regardless of paper color.
    """
    # Convert both to numpy arrays (float for multiplication)
    orig_array = np.array(original.convert('RGB')).astype(np.float32) / 255.0
    lines_array = np.array(lines.convert('RGB')).astype(np.float32) / 255.0

    # Multiply blend - darkens where lines are darker
    # This ensures lines show through even on colored paper
    blended = orig_array * lines_array

    # Convert back to uint8
    blended = (blended * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(blended, mode='RGB')


def add_lines_to_image(image_path, output_path, realistic=True):
    """Add ruled lines to a single image and save it."""
    # Load the original image
    original = Image.open(image_path)

    # Generate ruler with random parameters matching image size
    params = generate_random_ruler_params(original.size, realistic=realistic)
    ruler = RulerGenerator(**params)

    # Blend the images
    result = blend_images(original, ruler.image)

    # Save the result
    result.save(output_path, quality=95)

    return result


def main(limit=None, repeat=1, realistic=True, mix=False, val_split=0.2):
    """Process all unruled images and add lines to them.

    Args:
        limit: If specified, only process this many images (for testing)
        repeat: Number of variations to create for each image (skips if already have this many)
        realistic: If True, generate lines matching real ruled paper
        mix: If True, generate 50% realistic and 50% varied lines
        val_split: Fraction of parent images to use for validation (default 0.2)
    """
    import random

    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    # Get all image files from unruled directory
    image_files = sorted([
        f for f in os.listdir(UNRULED_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if limit:
        # Randomly select images when limit is specified
        image_files = random.sample(image_files, min(limit, len(image_files)))

    # Split parent images into train/validation sets
    random.shuffle(image_files)
    val_count = int(len(image_files) * val_split)
    val_files = set(image_files[:val_count])
    train_files = image_files[val_count:]

    print(f"Found {len(image_files)} source images")
    print(f"Training parents: {len(train_files)} ({100*(1-val_split):.0f}%)")
    print(f"Validation parents: {len(val_files)} ({100*val_split:.0f}%)")
    print(f"Target: {repeat} variation(s) per image")

    # Calculate how many new images we actually need to create
    train_needed = 0
    val_needed = 0
    for filename in image_files:
        name, _ = os.path.splitext(filename)
        if filename in val_files:
            existing = count_existing_variations(VALIDATION_DIR, name)
            val_needed += max(0, repeat - existing)
        else:
            existing = count_existing_variations(OUTPUT_DIR, name)
            train_needed += max(0, repeat - existing)

    total_needed = train_needed + val_needed
    print(f"Creating {train_needed} training images + {val_needed} validation images = {total_needed} total")

    output_count = 0
    for filename in image_files:
        input_path = os.path.join(UNRULED_DIR, filename)
        name, ext = os.path.splitext(filename)

        # Determine output directory based on parent image assignment
        is_validation = filename in val_files
        target_dir = VALIDATION_DIR if is_validation else OUTPUT_DIR
        set_name = "val" if is_validation else "train"

        # Check how many variations already exist
        existing_count = count_existing_variations(target_dir, name)
        needed = max(0, repeat - existing_count)

        if needed == 0:
            continue  # Skip - already have enough variations

        for r in range(needed):
            # Create output filename with per-image suffix
            suffix = get_next_suffix_for_image(target_dir, name)
            output_filename = f"{name}_{suffix:05d}{ext}"
            output_path = os.path.join(target_dir, output_filename)

            try:
                # Determine if this image should use realistic or varied lines
                use_realistic = realistic
                if mix:
                    use_realistic = np.random.random() < 0.5

                add_lines_to_image(input_path, output_path, realistic=use_realistic)
                output_count += 1
                mode_str = "realistic" if use_realistic else "varied"
                print(f"[{output_count}/{total_needed}] ({set_name}) Processed ({mode_str}): {filename} -> {output_filename}")
            except Exception as e:
                print(f"[{output_count}/{total_needed}] Error processing {filename}: {e}")

    print(f"\nDone! Created {output_count} new images.")
    print(f"Training images saved to: {OUTPUT_DIR}")
    print(f"Validation images saved to: {VALIDATION_DIR}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add ruled lines to unruled sketches')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to process (for testing)')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of variations to create per image (default: 1)')
    parser.add_argument('--varied', action='store_true',
                        help='Use varied synthetic lines instead of realistic ones')
    parser.add_argument('--mix', action='store_true',
                        help='Mix 50%% realistic and 50%% varied lines')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of parent images for validation (default: 0.2)')
    args = parser.parse_args()
    main(limit=args.limit, repeat=args.repeat, realistic=not args.varied, mix=args.mix,
         val_split=args.val_split)
