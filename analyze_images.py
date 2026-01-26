#!/usr/bin/env python3
"""
Analyze images in lines_added directory and create a DataFrame with metadata.
"""

import os
import re
import hashlib
import pandas as pd
from PIL import Image

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Sketches')
LINES_ADDED_DIR = os.path.join(DATA_DIR, 'lines_added')
UNRULED_DIR = os.path.join(DATA_DIR, 'Unruled')


def get_base_name(filename):
    """Extract base name from a filename by removing the suffix like _00001."""
    name_without_ext = os.path.splitext(filename)[0]
    match = re.match(r'(.+)_\d{5}$', name_without_ext)
    if match:
        return match.group(1)
    return name_without_ext


def find_parent_image(lined_filename):
    """Find the corresponding parent image in Unruled directory."""
    base_name = get_base_name(lined_filename)

    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        parent_path = os.path.join(UNRULED_DIR, base_name + ext)
        if os.path.exists(parent_path):
            return base_name + ext
    return None


def get_file_hash(filepath):
    """Calculate MD5 hash of file contents."""
    hash_md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def analyze_images():
    """Analyze all images in lines_added directory."""
    # Get all image files
    image_files = sorted([
        f for f in os.listdir(LINES_ADDED_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"Analyzing {len(image_files)} images...")

    data = []
    for i, filename in enumerate(image_files):
        filepath = os.path.join(LINES_ADDED_DIR, filename)

        try:
            # Get file size
            file_size = os.path.getsize(filepath)

            # Get parent image
            parent_image = find_parent_image(filename)

            # Get file hash
            file_hash = get_file_hash(filepath)

            # Get image dimensions and format
            with Image.open(filepath) as img:
                width, height = img.size
                image_format = img.format

            data.append({
                'filename': filename,
                'file_size': file_size,
                'parent_image': parent_image,
                'file_hash': file_hash,
                'width': width,
                'height': height,
                'image_format': image_format
            })

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(image_files)}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    df = pd.DataFrame(data)
    print(f"\nCreated DataFrame with {len(df)} rows")

    return df


def main():
    df = analyze_images()

    # Print summary
    print("\n=== Summary ===")
    print(f"Total images: {len(df)}")
    print(f"Unique parent images: {df['parent_image'].nunique()}")
    print(f"Images without parent: {df['parent_image'].isna().sum()}")
    print(f"Unique file hashes: {df['file_hash'].nunique()}")

    # Check for duplicate hashes (potential duplicate images)
    duplicate_hashes = df[df.duplicated(subset=['file_hash'], keep=False)]
    if len(duplicate_hashes) > 0:
        print(f"\nDuplicate content detected: {len(duplicate_hashes)} files with duplicate hashes")
        print("Hash counts:")
        print(duplicate_hashes['file_hash'].value_counts().head(10))

    print("\nImage dimensions:")
    print(df[['width', 'height']].describe())

    print("\nImage formats:")
    print(df['image_format'].value_counts())

    # Save to CSV
    output_path = os.path.join(BASE_DIR, 'image_analysis.csv')
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return df


if __name__ == '__main__':
    main()
