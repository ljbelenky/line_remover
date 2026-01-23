#!/usr/bin/env python3
"""
Line Remover - A TensorFlow CNN to remove ruled lines from sketch images.

Uses a denoising encoder/decoder architecture where:
- Input: Image with ruled lines (from lines_added directory)
- Target: Original image without lines (from Unruled directory)
"""

import os
import re
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Sketches')
LINES_ADDED_DIR = os.path.join(DATA_DIR, 'lines_added')
UNRULED_DIR = os.path.join(DATA_DIR, 'Unruled')
LINES_REMOVED_DIR = os.path.join(DATA_DIR, 'lines_removed')
PROGRESS_DIR = os.path.join(BASE_DIR, 'progress')
MODEL_PATH = os.path.join(BASE_DIR, 'line_remover_model.keras')

# Model parameters
IMG_HEIGHT = 512
IMG_WIDTH = 512
BATCH_SIZE = 2  # With bfloat16 mixed precision on 4GB GPU
EPOCHS = 200
PATIENCE = 25
MAX_TRAINING_TIME = 5 * 60  # 5 minutes in seconds


class TimeLimitCallback(keras.callbacks.Callback):
    """Stop training after a specified time limit."""

    def __init__(self, max_time_seconds):
        super().__init__()
        self.max_time_seconds = max_time_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_time_seconds:
            print(f"\nTime limit reached ({self.max_time_seconds}s). Stopping training.")
            self.model.stop_training = True


class ProgressCallback(keras.callbacks.Callback):
    """Save comparison images when loss improves."""

    def __init__(self, sample_image_path):
        super().__init__()
        self.sample_image_path = sample_image_path
        self.best_loss = float('inf')
        os.makedirs(PROGRESS_DIR, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss is not None and current_loss < self.best_loss:
            self.best_loss = current_loss
            self._save_comparison(current_loss)

    def _save_comparison(self, loss):
        """Generate and save a comparison image."""
        # Load sample image
        original_img = Image.open(self.sample_image_path).convert('RGB')
        original_size = original_img.size

        # Preprocess for model
        img = original_img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        result = self.model.predict(img_array, verbose=0)[0]

        # Convert back to image
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(result)
        result_img = result_img.resize(original_size, Image.Resampling.LANCZOS)

        # Create side-by-side comparison
        width, height = original_img.size
        comparison = Image.new('RGB', (width * 2, height))
        comparison.paste(original_img, (0, 0))
        comparison.paste(result_img, (width, 0))

        # Save with loss as filename
        filename = f"{loss:.6f}.jpg"
        output_path = os.path.join(PROGRESS_DIR, filename)
        comparison.save(output_path, quality=95)
        print(f" -> Saved progress image: {filename}")


def get_base_name(filename):
    """Extract base name from a filename by removing the suffix like _00001."""
    name_without_ext = os.path.splitext(filename)[0]
    # Remove the trailing _XXXXX suffix (5 digits)
    match = re.match(r'(.+)_\d{5}$', name_without_ext)
    if match:
        return match.group(1)
    return name_without_ext


def find_original_image(lined_filename):
    """Find the corresponding original image in Unruled directory."""
    base_name = get_base_name(lined_filename)

    # Try common extensions
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        original_path = os.path.join(UNRULED_DIR, base_name + ext)
        if os.path.exists(original_path):
            return original_path

    return None


def load_image_tf(path):
    """Load and preprocess an image using TensorFlow ops."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method='lanczos3')
    img = tf.cast(img, tf.float32) / 255.0
    return img


def load_image_pair(lined_path, original_path):
    """Load a pair of images (lined and original)."""
    lined_img = load_image_tf(lined_path)
    original_img = load_image_tf(original_path)
    return lined_img, original_img


def create_dataset():
    """Create training dataset from lines_added and Unruled directories."""
    # Get all files in lines_added directory
    lined_files = [f for f in os.listdir(LINES_ADDED_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(lined_files)} images in lines_added directory")

    # Build list of valid file path pairs
    lined_paths = []
    original_paths = []

    for lined_file in lined_files:
        lined_path = os.path.join(LINES_ADDED_DIR, lined_file)
        original_path = find_original_image(lined_file)

        if original_path is None:
            print(f"Warning: Could not find original for {lined_file}")
            continue

        lined_paths.append(lined_path)
        original_paths.append(original_path)

    num_samples = len(lined_paths)
    print(f"Found {num_samples} valid image pairs")

    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((lined_paths, original_paths))

    # Load images in parallel using all CPU cores
    dataset = dataset.map(
        load_image_pair,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False  # Allow out-of-order for speed
    )

    # Shuffle after loading - buffer contains actual images for better mixing
    # Using buffer_size = num_samples ensures full shuffle each epoch
    dataset = dataset.shuffle(buffer_size=num_samples, reshuffle_each_iteration=True)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, num_samples


def build_model():
    """
    Build a denoising encoder/decoder CNN.

    Architecture uses skip connections (U-Net style) for better detail preservation.
    4-level U-Net for 512x512 input: 512→256→128→64→32 (bottleneck)
    Target: < 10,000,000 parameters
    """
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Encoder
    # Block 1: 512x512 -> 256x256
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip1 = x
    x = layers.MaxPooling2D(2)(x)

    # Block 2: 256x256 -> 128x128
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip2 = x
    x = layers.MaxPooling2D(2)(x)

    # Block 3: 128x128 -> 64x64
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip3 = x
    x = layers.MaxPooling2D(2)(x)

    # Block 4: 64x64 -> 32x32
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip4 = x
    x = layers.MaxPooling2D(2)(x)

    # Bottleneck: 32x32
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Decoder
    # Block 4: 32x32 -> 64x64
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip4])
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 3: 64x64 -> 128x128
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 2: 128x128 -> 256x256
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 1: 256x256 -> 512x512
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Output layer - linear activation, clip values in post-processing
    # Use float32 for output layer for numerical stability with mixed precision
    x = layers.Conv2D(3, 1, padding='same', activation='linear')(x)
    outputs = layers.Activation('linear', dtype='float32')(x)

    model = keras.Model(inputs, outputs, name='line_remover')

    return model


def train_model(model, dataset, num_samples, sample_image_path=None):
    """Train the model."""
    # Compile with MSE loss (good for image reconstruction)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )

    # Callbacks
    callbacks = [
        TimeLimitCallback(MAX_TRAINING_TIME),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss', patience=PATIENCE, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            save_best_only=True, monitor='loss', verbose=1
        )
    ]

    # Add progress callback if sample image provided
    if sample_image_path:
        callbacks.append(ProgressCallback(sample_image_path))

    # Train
    steps_per_epoch = max(1, num_samples // BATCH_SIZE)
    history = model.fit(
        dataset.repeat(),
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )

    return history


def remove_lines(model, image_path, output_path=None):
    """Use the trained model to remove lines from an image."""
    # Load and preprocess
    original_img = Image.open(image_path).convert('RGB')
    original_size = original_img.size

    img = original_img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    result = model.predict(img_array, verbose=0)[0]

    # Convert back to image
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result)

    # Resize back to original size
    result_img = result_img.resize(original_size, Image.Resampling.LANCZOS)

    if output_path:
        result_img.save(output_path, quality=95)

    return result_img


def create_comparison_image(input_img, output_img):
    """Create a side-by-side comparison image with input on left, output on right."""
    # Ensure both images are the same size
    width, height = input_img.size

    # Create new image with double width
    comparison = Image.new('RGB', (width * 2, height))

    # Paste input on left, output on right
    comparison.paste(input_img, (0, 0))
    comparison.paste(output_img, (width, 0))

    return comparison


def process_all_images(model, limit=None, compare=False):
    """Process all images from lines_added and save to lines_removed.

    Args:
        model: The trained model
        limit: If specified, only process this many images
        compare: If True, create side-by-side comparison images (input | output)
    """
    os.makedirs(LINES_REMOVED_DIR, exist_ok=True)

    # Get all image files
    image_files = sorted([
        f for f in os.listdir(LINES_ADDED_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if limit:
        image_files = image_files[:limit]

    mode_str = "comparison" if compare else "output"
    print(f"Processing {len(image_files)} images ({mode_str} mode)...")

    for i, filename in enumerate(image_files):
        input_path = os.path.join(LINES_ADDED_DIR, filename)
        output_path = os.path.join(LINES_REMOVED_DIR, filename)

        try:
            # Get the output image
            result_img = remove_lines(model, input_path)

            if compare:
                # Create side-by-side comparison
                input_img = Image.open(input_path).convert('RGB')
                comparison = create_comparison_image(input_img, result_img)
                comparison.save(output_path, quality=95)
            else:
                # Save just the output
                result_img.save(output_path, quality=95)

            print(f"[{i+1}/{len(image_files)}] Processed: {filename}")
        except Exception as e:
            print(f"[{i+1}/{len(image_files)}] Error processing {filename}: {e}")

    print(f"\nDone! Processed {len(image_files)} images.")
    print(f"Output saved to: {LINES_REMOVED_DIR}")


def train(resume=False):
    """Train the model.

    Args:
        resume: If True, continue training from saved model. If False, start fresh.
    """
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) available: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Enable mixed precision to reduce memory usage
        # bfloat16 has same range as float32, more stable than float16
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
        print("Mixed precision (bfloat16) enabled for reduced memory usage")
    else:
        print("No GPU available, using CPU")

    # Create dataset
    print("\nLoading dataset...")
    dataset, num_samples = create_dataset()

    if num_samples == 0:
        print("Error: No valid image pairs found. Please run add_lines.py first.")
        return

    # Get a sample image for progress visualization
    sample_files = [f for f in os.listdir(LINES_ADDED_DIR)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sample_image_path = os.path.join(LINES_ADDED_DIR, sample_files[0]) if sample_files else None

    # Build or load model
    if resume and os.path.exists(MODEL_PATH):
        print(f"\nResuming training from {MODEL_PATH}...")
        model = keras.models.load_model(MODEL_PATH)
    else:
        if resume:
            print(f"\nNo saved model found at {MODEL_PATH}, starting fresh...")
        print("\nBuilding model...")
        model = build_model()
    model.summary()

    # Check parameter count
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    if total_params > 10_000_000:
        print("Warning: Model exceeds 10,000,000 parameter limit")
    else:
        print("Model is within parameter limit")

    # Train
    print("\nStarting training...")
    train_model(model, dataset, num_samples, sample_image_path)

    print("\nTraining complete!")
    print(f"Model saved to: {MODEL_PATH}")


def inference(limit=None, compare=False):
    """Load trained model and process images."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first with: python line_remover.py --train")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)

    process_all_images(model, limit=limit, compare=compare)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Remove ruled lines from sketch images')
    parser.add_argument('--train', action='store_true',
                        help='Train the model from scratch')
    parser.add_argument('--resume', action='store_true',
                        help='Continue training from saved model')
    parser.add_argument('--process', action='store_true',
                        help='Process images from lines_added to lines_removed')
    parser.add_argument('--compare', action='store_true',
                        help='Create side-by-side comparison images (input | output)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to process')
    args = parser.parse_args()

    if args.train or args.resume:
        train(resume=args.resume)
    elif args.process or args.compare:
        inference(limit=args.limit, compare=args.compare)
    else:
        print("Usage:")
        print("  python line_remover.py --train              Train the model from scratch")
        print("  python line_remover.py --resume             Continue training from saved model")
        print("  python line_remover.py --process            Process all images")
        print("  python line_remover.py --compare            Create side-by-side comparisons")
        print("  python line_remover.py --process --limit 5  Process 5 images")


if __name__ == '__main__':
    main()

