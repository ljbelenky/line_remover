#!/usr/bin/env python3
"""
Display input/target patch pairs side-by-side for visual comparison.
Press Enter to advance, 'q' to quit.
"""

import os
import cv2
import numpy as np

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PATCHES_DIR = os.path.join(DATA_DIR, 'patches')
TRAIN_INPUT_DIR = os.path.join(PATCHES_DIR, 'train', 'input')
TRAIN_TARGET_DIR = os.path.join(PATCHES_DIR, 'train', 'target')


def get_target_filename(input_filename):
    """Extract target filename from input filename.

    Input: IMG_0020_00001_p000_00.jpg
    Target: IMG_0020_00001_p000.jpg
    """
    name, ext = os.path.splitext(input_filename)
    # Remove the last underscore and suffix (e.g., _00, _01)
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0] + ext
    return input_filename


def compare_pairs(input_dir=TRAIN_INPUT_DIR, target_dir=TRAIN_TARGET_DIR):
    """Display input/target pairs side by side."""

    # Get input files
    input_files = sorted([f for f in os.listdir(input_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not input_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(input_files)} input images")
    print("Press Enter to advance, 'q' to quit\n")

    for i, input_file in enumerate(input_files):
        input_path = os.path.join(input_dir, input_file)
        target_file = get_target_filename(input_file)
        target_path = os.path.join(target_dir, target_file)

        # Load images
        input_img = cv2.imread(input_path)
        if input_img is None:
            print(f"Could not load input: {input_file}")
            continue

        if os.path.exists(target_path):
            target_img = cv2.imread(target_path)
        else:
            print(f"Target not found: {target_file}")
            target_img = np.zeros_like(input_img)

        # Create side-by-side comparison
        comparison = np.hstack([input_img, target_img])

        # Add labels
        h, w = input_img.shape[:2]
        cv2.putText(comparison, "INPUT (with lines)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(comparison, "INPUT (with lines)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(comparison, "TARGET (clean)", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(comparison, "TARGET (clean)", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        # Show filename and progress
        info = f"[{i+1}/{len(input_files)}] {input_file}"
        cv2.putText(comparison, info, (10, comparison.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(comparison, info, (10, comparison.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow('Input vs Target Comparison', comparison)

        # Wait for key
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            print("Quitting...")
            break

    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    compare_pairs()
