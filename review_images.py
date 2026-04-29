#!/usr/bin/env python3
"""
Image review tool for manually curating patches.
Press 'a' to accept (keep) or 'd' to delete.
Press 'q' to quit and save progress.
"""

import os
import cv2
import sys

def review_images(directory):
    """Review images in directory, allowing user to accept or delete each."""

    # Get list of images
    images = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total = len(images)

    if total == 0:
        print(f"No images found in {directory}")
        return

    print(f"Found {total} images to review")
    print("Controls:")
    print("  'a' or RIGHT ARROW - Accept (keep image)")
    print("  'd' or LEFT ARROW  - Delete image")
    print("  'q' or ESC         - Quit")
    print()

    deleted = 0
    accepted = 0
    i = 0

    while i < len(images):
        filename = images[i]
        filepath = os.path.join(directory, filename)

        # Check if file still exists (might have been deleted)
        if not os.path.exists(filepath):
            i += 1
            continue

        # Load and display image
        img = cv2.imread(filepath)
        if img is None:
            print(f"Could not load: {filename}")
            i += 1
            continue

        # Scale image to fit screen
        screen_w, screen_h = 1920, 1080
        img_h, img_w = img.shape[:2]
        scale = min(screen_w / img_w, (screen_h - 80) / img_h, 1.0)
        if scale < 1.0:
            display_img = cv2.resize(img, (int(img_w * scale), int(img_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            display_img = img.copy()

        # Add text overlay with progress info
        text = f"[{i+1}/{total}] {filename} (a=accept, d=delete, q=quit)"
        cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        stats = f"Accepted: {accepted} | Deleted: {deleted}"
        cv2.putText(display_img, stats, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(display_img, stats, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow('Image Review', display_img)

        # Wait for keypress
        key = cv2.waitKey(0) & 0xFF

        if key == ord('a') or key == 83:  # 'a' or right arrow
            accepted += 1
            i += 1
        elif key == ord('d') or key == 81:  # 'd' or left arrow
            os.remove(filepath)
            deleted += 1
            images.pop(i)  # Remove from list
            print(f"Deleted: {filename}")
        elif key == ord('q') or key == 27:  # 'q' or ESC
            print("\nQuitting...")
            break

    cv2.destroyAllWindows()

    print(f"\nReview complete!")
    print(f"  Accepted: {accepted}")
    print(f"  Deleted: {deleted}")
    print(f"  Remaining: {len([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = 'results/compare'

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        sys.exit(1)

    review_images(directory)
