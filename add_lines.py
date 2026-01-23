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


def generate_random_ruler_params(shape):
    """Generate randomized parameters for ruler lines."""
    return {
        'shape': shape,
        'line_width': np.clip(np.random.randint(-2, 4), 1, 4),
        'lines': np.random.randint(15, 40),
        'v_offset': np.random.random(),
        'raggedness': -0.15 + np.random.random() / 2,
        'color': np.random.randint(100, 175),
        'color_variation': np.random.randint(-15, 16),
        'angle': np.random.randint(-7, 8)
    }


def blend_images(original, lines):
    """Blend the ruled lines with the original image using minimum (darken) blend."""
    # Convert both to numpy arrays
    orig_array = np.array(original.convert('RGB'))
    lines_array = np.array(lines.convert('RGB'))

    # Use minimum blend - takes the darker pixel value
    # This makes the lines appear on top of the sketch
    blended = np.minimum(orig_array, lines_array)

    return Image.fromarray(blended, mode='RGB')


def add_lines_to_image(image_path, output_path):
    """Add ruled lines to a single image and save it."""
    # Load the original image
    original = Image.open(image_path)

    # Generate ruler with random parameters matching image size
    params = generate_random_ruler_params(original.size)
    ruler = RulerGenerator(**params)

    # Blend the images
    result = blend_images(original, ruler.image)

    # Save the result
    result.save(output_path, quality=95)

    return result


def main(limit=None, repeat=1):
    """Process all unruled images and add lines to them.

    Args:
        limit: If specified, only process this many images (for testing)
        repeat: Number of variations to create for each image
    """
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all image files from unruled directory
    image_files = sorted([
        f for f in os.listdir(UNRULED_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if limit:
        image_files = image_files[:limit]

    total_outputs = len(image_files) * repeat
    print(f"Found {len(image_files)} images to process")
    print(f"Creating {repeat} variation(s) per image ({total_outputs} total outputs)")

    output_count = 0
    for i, filename in enumerate(image_files):
        input_path = os.path.join(UNRULED_DIR, filename)
        name, ext = os.path.splitext(filename)

        for r in range(repeat):
            # Create output filename with per-image suffix
            suffix = get_next_suffix_for_image(OUTPUT_DIR, name)
            output_filename = f"{name}_{suffix:05d}{ext}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            try:
                add_lines_to_image(input_path, output_path)
                output_count += 1
                print(f"[{output_count}/{total_outputs}] Processed: {filename} -> {output_filename}")
            except Exception as e:
                print(f"[{output_count}/{total_outputs}] Error processing {filename}: {e}")

    print(f"\nDone! Created {output_count} images from {len(image_files)} source images.")
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add ruled lines to unruled sketches')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to process (for testing)')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of variations to create per image (default: 1)')
    args = parser.parse_args()
    main(limit=args.limit, repeat=args.repeat)
