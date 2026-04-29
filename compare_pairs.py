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


def draw_histogram(input_img, target_img, width, height=200):
    """Draw pixel brightness histograms for input and target images.

    Returns a BGR image of the histogram plot.
    """
    hist_img = np.ones((height, width, 3), dtype=np.uint8) * 240  # light gray bg

    # Convert to grayscale brightness
    input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # Compute histograms (256 bins, range 0-255)
    input_hist = cv2.calcHist([input_gray], [0], None, [256], [0, 256]).flatten()
    target_hist = cv2.calcHist([target_gray], [0], None, [256], [0, 256]).flatten()

    # Normalize to fit in histogram height using log scale
    margin_top = 25
    margin_bottom = 25
    plot_height = height - margin_top - margin_bottom

    # Apply log scale: log(1 + count) to handle zeros
    max_raw = max(input_hist.max(), target_hist.max(), 1)
    input_log = np.log1p(input_hist)
    target_log = np.log1p(target_hist)
    max_log = max(input_log.max(), target_log.max(), 1)
    input_hist = (input_log / max_log * plot_height).astype(np.int32)
    target_hist = (target_log / max_log * plot_height).astype(np.int32)

    # Draw horizontal grid lines at log-spaced values
    for power in [1, 10, 100, 1000, 10000]:
        if power > max_raw:
            break
        frac = np.log1p(power) / max_log
        y = margin_top + int(plot_height * (1 - frac))
        cv2.line(hist_img, (0, y), (width, y), (210, 210, 210), 1)
        cv2.putText(hist_img, str(power), (width - 45, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    # Draw histograms as filled areas
    x_scale = width / 256.0
    for b in range(256):
        x = int(b * x_scale)
        x_next = int((b + 1) * x_scale)

        # Target histogram (green, drawn first)
        if target_hist[b] > 0:
            y_top = margin_top + plot_height - target_hist[b]
            y_bottom = margin_top + plot_height
            cv2.rectangle(hist_img, (x, y_top), (x_next, y_bottom), (0, 180, 0), -1)

        # Input histogram (blue, semi-transparent via drawing second)
        if input_hist[b] > 0:
            y_top = margin_top + plot_height - input_hist[b]
            y_bottom = margin_top + plot_height
            overlay = hist_img.copy()
            cv2.rectangle(overlay, (x, y_top), (x_next, y_bottom), (200, 100, 0), -1)
            cv2.addWeighted(overlay, 0.5, hist_img, 0.5, 0, hist_img)

    # Draw axis line
    cv2.line(hist_img, (0, margin_top + plot_height), (width, margin_top + plot_height), (0, 0, 0), 1)

    # Labels
    cv2.putText(hist_img, "Pixel Brightness Histogram", (10, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(hist_img, "Input", (width - 200, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 0), 1)
    cv2.putText(hist_img, "Target", (width - 110, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)

    # X-axis labels (brightness values)
    for val in [0, 64, 128, 192, 255]:
        x = int(val * x_scale)
        cv2.putText(hist_img, str(val), (x, height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # Mean brightness annotations
    input_mean = input_gray.mean()
    target_mean = target_gray.mean()
    cv2.putText(hist_img, f"Input mean: {input_mean:.1f}", (10, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 100, 0), 1)
    cv2.putText(hist_img, f"Target mean: {target_mean:.1f}", (150, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 0), 1)

    return hist_img


def compare_pairs(input_dir=TRAIN_INPUT_DIR, target_dir=TRAIN_TARGET_DIR):
    """Display input/target pairs side by side with brightness histogram."""

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

        # Draw brightness histogram and stack below comparison
        hist_img = draw_histogram(input_img, target_img, comparison.shape[1])
        display = np.vstack([comparison, hist_img])

        cv2.imshow('Input vs Target Comparison', display)

        # Wait for key
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            print("Quitting...")
            break

    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    compare_pairs()
