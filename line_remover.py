#!/usr/bin/env python3
"""
Line Remover - A TensorFlow CNN to remove ruled lines from sketch images.

Uses a denoising encoder/decoder architecture where:
- Input: Image with ruled lines (from lines_added directory)
- Target: Original image without lines (from Unruled directory)
"""

import os
import random
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
RULED_DIR = os.path.join(DATA_DIR, 'ruled_rgb')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PROGRESS_DIR = os.path.join(BASE_DIR, 'progress')
MODEL_PATH = os.path.join(BASE_DIR, 'line_remover_model.keras')

# Model parameters
PATCH_SIZE = 512  # Train on random patches at original resolution
BATCH_SIZE = 2  # Reduced for larger patches
EPOCHS = 2000
PATIENCE = 50
MAX_TRAINING_TIME = 6*3600  # in seconds
PATCHES_PER_EPOCH = 1000  # Random patches per epoch
MIN_IMAGE_SIZE = 512  # Skip images smaller than patch size

# Inference parameters
TILE_SIZE = 512  # Process tiles at this size
TILE_OVERLAP = 64  # Overlap between tiles for blending


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

    def __init__(self, sample_image_path, target_image_path=None, ruled_image_path=None):
        super().__init__()
        self.sample_image_path = sample_image_path
        self.target_image_path = target_image_path
        self.ruled_image_path = ruled_image_path
        self.best_loss = float('inf')
        self.current_epoch = 0
        # Extract base filename for naming
        self.sample_name = os.path.splitext(os.path.basename(sample_image_path))[0]
        if ruled_image_path:
            self.ruled_name = os.path.splitext(os.path.basename(ruled_image_path))[0]
        os.makedirs(PROGRESS_DIR, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        current_loss = logs.get('loss')
        if current_loss is not None and current_loss < self.best_loss:
            self.best_loss = current_loss
            self._save_comparison(current_loss)
            self._save_ruled_comparison(current_loss)

    def _save_comparison(self, loss):
        """Generate and save a 4-panel comparison image using tile-based inference."""
        # Load input image (from lines_added)
        input_img = Image.open(self.sample_image_path).convert('RGB')

        # Use tile-based inference
        output_img = remove_lines_tiled(self.model, input_img)

        # Load target image (from Unruled) if available
        target_img = None
        if self.target_image_path and os.path.exists(self.target_image_path):
            target_img = Image.open(self.target_image_path).convert('RGB')

        # Create comparison image (4-panel if target exists, 2-panel otherwise)
        comparison = create_comparison_image(input_img, output_img, target_img)

        # Save with filename format: {image_name}_epoch{epoch}_{loss}.jpg
        filename = f"{self.sample_name}_epoch{self.current_epoch:04d}_{loss:.6f}.jpg"
        output_path = os.path.join(PROGRESS_DIR, filename)
        comparison.save(output_path, quality=95)
        print(f" -> Saved progress image: {filename}")

    def _save_ruled_comparison(self, loss):
        """Generate and save a 2-panel comparison for a real ruled paper image."""
        if not self.ruled_image_path or not os.path.exists(self.ruled_image_path):
            return

        # Load ruled image
        input_img = Image.open(self.ruled_image_path).convert('RGB')

        # Use tile-based inference
        output_img = remove_lines_tiled(self.model, input_img)

        # Create 2-panel comparison (no target for ruled images)
        comparison = create_comparison_image(input_img, output_img)

        # Save with filename format: ruled_{image_name}_epoch{epoch}_{loss}.jpg
        filename = f"ruled_{self.ruled_name}_epoch{self.current_epoch:04d}_{loss:.6f}.jpg"
        output_path = os.path.join(PROGRESS_DIR, filename)
        comparison.save(output_path, quality=95)
        print(f" -> Saved ruled progress image: {filename}")


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
    """Load an image at original resolution using TensorFlow ops."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def extract_random_patch(lined_img, original_img):
    """Extract a random patch from the same location in both images."""
    shape = tf.shape(lined_img)
    height, width = shape[0], shape[1]

    # Random top-left corner for the patch
    max_y = height - PATCH_SIZE
    max_x = width - PATCH_SIZE
    y = tf.random.uniform([], 0, max_y + 1, dtype=tf.int32)
    x = tf.random.uniform([], 0, max_x + 1, dtype=tf.int32)

    # Extract patches from same location
    lined_patch = lined_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]
    original_patch = original_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]

    # Ensure correct shape
    lined_patch = tf.ensure_shape(lined_patch, [PATCH_SIZE, PATCH_SIZE, 3])
    original_patch = tf.ensure_shape(original_patch, [PATCH_SIZE, PATCH_SIZE, 3])

    return lined_patch, original_patch


def load_image_pair(lined_path, original_path):
    """Load a pair of images and extract a random patch."""
    lined_img = load_image_tf(lined_path)
    original_img = load_image_tf(original_path)

    # Resize original to match lined if different sizes
    lined_shape = tf.shape(lined_img)
    original_img = tf.image.resize(original_img, [lined_shape[0], lined_shape[1]])

    return extract_random_patch(lined_img, original_img)


def get_image_pairs():
    """Get all valid image path pairs from lines_added and Unruled directories.

    Filters out images smaller than MIN_IMAGE_SIZE.
    """
    lined_files = [f for f in os.listdir(LINES_ADDED_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(lined_files)} images in lines_added directory")

    lined_paths = []
    original_paths = []
    skipped_small = 0

    for lined_file in lined_files:
        lined_path = os.path.join(LINES_ADDED_DIR, lined_file)
        original_path = find_original_image(lined_file)

        if original_path is None:
            continue

        # Check image size - skip if too small for patches
        try:
            with Image.open(lined_path) as img:
                width, height = img.size
                if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                    skipped_small += 1
                    continue
        except Exception:
            continue

        lined_paths.append(lined_path)
        original_paths.append(original_path)

    num_samples = len(lined_paths)
    print(f"Found {num_samples} valid image pairs (skipped {skipped_small} too small)")

    return lined_paths, original_paths, num_samples


def get_validation_pairs():
    """Get validation image path pairs from validation directory.

    Filters out images smaller than MIN_IMAGE_SIZE.
    """
    if not os.path.exists(VALIDATION_DIR):
        print("No validation directory found")
        return [], [], 0

    val_files = [f for f in os.listdir(VALIDATION_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(val_files)} images in validation directory")

    lined_paths = []
    original_paths = []
    skipped_small = 0

    for val_file in val_files:
        val_path = os.path.join(VALIDATION_DIR, val_file)
        original_path = find_original_image(val_file)

        if original_path is None:
            continue

        # Check image size - skip if too small for patches
        try:
            with Image.open(val_path) as img:
                width, height = img.size
                if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                    skipped_small += 1
                    continue
        except Exception:
            continue

        lined_paths.append(val_path)
        original_paths.append(original_path)

    num_samples = len(lined_paths)
    print(f"Found {num_samples} valid validation pairs (skipped {skipped_small} too small)")

    return lined_paths, original_paths, num_samples


def create_dataset_from_paths(lined_paths, original_paths):
    """Create a tf.data.Dataset from lists of file paths."""
    dataset = tf.data.Dataset.from_tensor_slices((lined_paths, original_paths))
    dataset = dataset.map(
        load_image_pair,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def sample_random_patches(lined_paths, original_paths, n_patches):
    """Randomly sample image pairs for n_patches extractions.

    Since each image pair yields one random patch, we sample n_patches
    pairs with replacement to get variety.
    """
    n_images = len(lined_paths)
    indices = np.random.randint(0, n_images, size=n_patches)
    sampled_lined = [lined_paths[i] for i in indices]
    sampled_original = [original_paths[i] for i in indices]
    return sampled_lined, sampled_original


def build_model():
    """
    Build a denoising encoder/decoder CNN for patch-based training.

    Architecture uses skip connections (U-Net style) for better detail preservation.
    3-level U-Net for 256x256 patches: 256→128→64→32 (bottleneck)
    Target: < 10,000,000 parameters
    """
    inputs = keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, 3))

    # Encoder
    # Block 1: 256x256 -> 128x128
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip1 = x
    x = layers.MaxPooling2D(2)(x)

    # Block 2: 128x128 -> 64x64
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip2 = x
    x = layers.MaxPooling2D(2)(x)

    # Block 3: 64x64 -> 32x32
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip3 = x
    x = layers.MaxPooling2D(2)(x)

    # Bottleneck: 32x32
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Decoder
    # Block 3: 32x32 -> 64x64
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 2: 64x64 -> 128x128
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 1: 128x128 -> 256x256
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Output layer - linear activation, clip values in post-processing
    # Use float32 for output layer for numerical stability with mixed precision
    x = layers.Conv2D(3, 1, padding='same', activation='linear')(x)
    outputs = layers.Activation('linear', dtype='float32')(x)

    model = keras.Model(inputs, outputs, name='line_remover')

    return model


def train_model(model, lined_paths, original_paths, num_samples,
                val_lined_paths=None, val_original_paths=None, val_num_samples=0,
                sample_image_path=None, target_image_path=None, ruled_image_path=None):
    """Train the model with random patch sampling each epoch.

    Uses validation set for early stopping and model selection.
    """
    import time

    # Compile with MSE loss (good for image reconstruction)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )

    patches_per_epoch = min(PATCHES_PER_EPOCH, num_samples * 10)  # Allow multiple patches per image
    steps_per_epoch = max(1, patches_per_epoch // BATCH_SIZE)

    # Validation patches - use fewer for efficiency
    val_patches = min(200, val_num_samples * 4) if val_num_samples > 0 else 0
    use_validation = val_num_samples > 0 and val_lined_paths and val_original_paths

    print(f"Using {patches_per_epoch} random patches per epoch ({steps_per_epoch} steps)")
    print(f"Training images: {num_samples}")
    print(f"Validation images: {val_num_samples}")
    if use_validation:
        print(f"Validation patches per epoch: {val_patches}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")

    # Progress callback for visualization
    progress_callback = ProgressCallback(sample_image_path, target_image_path, ruled_image_path) if sample_image_path else None
    if progress_callback:
        progress_callback.set_model(model)

    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    current_lr = 1e-3

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Check time limit
        if time.time() - start_time >= MAX_TRAINING_TIME:
            print(f"\nTime limit reached ({MAX_TRAINING_TIME}s). Stopping training.")
            break

        # Sample random image pairs for patch extraction
        sampled_lined, sampled_original = sample_random_patches(
            lined_paths, original_paths, patches_per_epoch)
        dataset = create_dataset_from_paths(sampled_lined, sampled_original)

        # Train for one epoch
        history = model.fit(dataset, epochs=1, verbose=0)
        train_loss = history.history['loss'][0]
        train_mae = history.history['mae'][0]

        # Compute validation loss
        if use_validation:
            val_sampled_lined, val_sampled_original = sample_random_patches(
                val_lined_paths, val_original_paths, val_patches)
            val_dataset = create_dataset_from_paths(val_sampled_lined, val_sampled_original)
            val_results = model.evaluate(val_dataset, verbose=0)
            val_loss = val_results[0]
            val_mae = val_results[1]
            # Use validation loss for early stopping
            loss_for_stopping = val_loss
            loss_str = (f"Epoch {epoch+1}/{EPOCHS} - loss: {train_loss:.6f} - mae: {train_mae:.6f} - "
                        f"val_loss: {val_loss:.6f} - val_mae: {val_mae:.6f}")
        else:
            # Fall back to training loss if no validation
            loss_for_stopping = train_loss
            loss_str = f"Epoch {epoch+1}/{EPOCHS} - loss: {train_loss:.6f} - mae: {train_mae:.6f}"

        print(f"{loss_str} - {time.time()-epoch_start:.1f}s")

        # Check for improvement (using validation loss if available)
        if loss_for_stopping < best_loss:
            best_loss = loss_for_stopping
            patience_counter = 0
            model.save(MODEL_PATH)
            print(f" -> Model saved (val_loss: {loss_for_stopping:.6f})")

            # Save progress image
            if progress_callback:
                progress_callback.current_epoch = epoch + 1
                progress_callback.best_loss = loss_for_stopping
                progress_callback._save_comparison(loss_for_stopping)
                progress_callback._save_ruled_comparison(loss_for_stopping)
        else:
            patience_counter += 1

        # Reduce LR on plateau
        if patience_counter > 0 and patience_counter % 5 == 0:
            current_lr *= 0.5
            if current_lr >= 1e-6:
                model.optimizer.learning_rate.assign(current_lr)
                print(f" -> Reducing learning rate to {current_lr}")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {PATIENCE} epochs)")
            break

    return None


def create_blend_mask(tile_size, overlap):
    """Create a smooth blending mask for tile edges."""
    mask = np.ones((tile_size, tile_size), dtype=np.float32)

    # Create linear ramps for the overlap regions
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap)

        # Top edge
        mask[:overlap, :] *= ramp[:, np.newaxis]
        # Bottom edge
        mask[-overlap:, :] *= ramp[::-1, np.newaxis]
        # Left edge
        mask[:, :overlap] *= ramp[np.newaxis, :]
        # Right edge
        mask[:, -overlap:] *= ramp[::-1][np.newaxis, :]

    return mask


def remove_lines_tiled(model, img, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    """Process an image using overlapping tiles and blend the results.

    Args:
        model: The trained model
        img: PIL Image or path to image
        tile_size: Size of each tile (default: TILE_SIZE)
        overlap: Overlap between tiles (default: TILE_OVERLAP)

    Returns:
        PIL Image with lines removed at original resolution
    """
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')

    width, height = img.size
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Pad image to be divisible by (tile_size - overlap)
    stride = tile_size - overlap
    pad_h = (stride - (height % stride)) % stride
    pad_w = (stride - (width % stride)) % stride

    # Also ensure we have at least one full tile
    if height + pad_h < tile_size:
        pad_h = tile_size - height
    if width + pad_w < tile_size:
        pad_w = tile_size - width

    # Pad with reflection to avoid edge artifacts
    padded = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    padded_h, padded_w = padded.shape[:2]

    # Output accumulator and weight accumulator
    output = np.zeros_like(padded)
    weights = np.zeros((padded_h, padded_w), dtype=np.float32)

    # Create blend mask
    blend_mask = create_blend_mask(tile_size, overlap)

    # Process tiles
    for y in range(0, padded_h - tile_size + 1, stride):
        for x in range(0, padded_w - tile_size + 1, stride):
            # Extract tile
            tile = padded[y:y+tile_size, x:x+tile_size, :]
            tile_batch = np.expand_dims(tile, axis=0)

            # Process tile
            result = model.predict(tile_batch, verbose=0)[0]

            # Accumulate with blending
            output[y:y+tile_size, x:x+tile_size, :] += result * blend_mask[:, :, np.newaxis]
            weights[y:y+tile_size, x:x+tile_size] += blend_mask

    # Normalize by weights
    weights = np.maximum(weights, 1e-8)  # Avoid division by zero
    output = output / weights[:, :, np.newaxis]

    # Remove padding
    output = output[:height, :width, :]

    # Convert back to image
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(output)


def remove_lines(model, image_path, output_path=None):
    """Use the trained model to remove lines from an image using tile-based inference."""
    # Load image
    original_img = Image.open(image_path).convert('RGB')

    # Use tile-based inference for full resolution
    result_img = remove_lines_tiled(model, original_img)

    if output_path:
        result_img.save(output_path, quality=95)

    return result_img


def compute_difference_image(output_img, target_img):
    """Compute the absolute difference between output and target images.

    Returns an image where brighter pixels indicate larger differences.
    """
    output_arr = np.array(output_img, dtype=np.float32)
    target_arr = np.array(target_img, dtype=np.float32)

    # Compute absolute difference and scale for visibility
    diff = np.abs(output_arr - target_arr)
    # Scale up the difference to make it more visible (multiply by 2)
    diff = np.clip(diff * 2, 0, 255).astype(np.uint8)

    return Image.fromarray(diff)


def create_comparison_image(input_img, output_img, target_img=None):
    """Create a side-by-side comparison image.

    Args:
        input_img: The input image (with lines)
        output_img: The neural network output
        target_img: The target/ground truth image (optional, from Unruled)

    Returns:
        If target_img is provided: 5-panel image (input | output | target | output-target diff | input-output diff)
        If target_img is None: 3-panel image (input | output | input-output diff)
    """
    width, height = input_img.size

    # Resize output to match input if different
    if output_img.size != (width, height):
        output_img = output_img.resize((width, height), Image.Resampling.LANCZOS)

    # Compute what the model changed (input vs output difference)
    # This shows what was removed/modified by the model
    removed_diff = compute_difference_image(input_img, output_img)

    if target_img is not None:
        # 5-panel comparison: input | output | target | output-target diff | input-output diff
        if target_img.size != (width, height):
            target_img = target_img.resize((width, height), Image.Resampling.LANCZOS)

        # Compute error (output vs target difference)
        error_diff = compute_difference_image(output_img, target_img)

        comparison = Image.new('RGB', (width * 5, height))
        comparison.paste(input_img, (0, 0))
        comparison.paste(output_img, (width, 0))
        comparison.paste(target_img, (width * 2, 0))
        comparison.paste(error_diff, (width * 3, 0))
        comparison.paste(removed_diff, (width * 4, 0))
    else:
        # 3-panel comparison: input | output | input-output diff
        comparison = Image.new('RGB', (width * 3, height))
        comparison.paste(input_img, (0, 0))
        comparison.paste(output_img, (width, 0))
        comparison.paste(removed_diff, (width * 2, 0))

    return comparison


def process_all_images(model, limit=None, compare=False):
    """Process all images from lines_added and save to lines_removed.

    Args:
        model: The trained model
        limit: If specified, only process this many images
        compare: If True, create 4-panel comparison images (input | output | target | diff)
    """
    os.makedirs(LINES_REMOVED_DIR, exist_ok=True)

    # Get all image files
    image_files = sorted([
        f for f in os.listdir(LINES_ADDED_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if limit and limit < len(image_files):
        # Randomly select images when limit is specified
        image_files = random.sample(image_files, limit)

    mode_str = "4-panel comparison" if compare else "output"
    print(f"Processing {len(image_files)} images ({mode_str} mode)...")

    for i, filename in enumerate(image_files):
        input_path = os.path.join(LINES_ADDED_DIR, filename)
        output_path = os.path.join(LINES_REMOVED_DIR, filename)

        try:
            # Get the output image
            result_img = remove_lines(model, input_path)

            if compare:
                # Create 4-panel comparison: input | output | target | difference
                input_img = Image.open(input_path).convert('RGB')

                # Find the target image (original from Unruled)
                target_path = find_original_image(filename)
                target_img = None
                if target_path and os.path.exists(target_path):
                    target_img = Image.open(target_path).convert('RGB')

                comparison = create_comparison_image(input_img, result_img, target_img)
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

    # Get training image pairs
    print("\nScanning for training image pairs...")
    lined_paths, original_paths, num_samples = get_image_pairs()

    if num_samples == 0:
        print("Error: No valid image pairs found. Please run add_lines.py first.")
        return

    # Get validation image pairs
    print("\nScanning for validation image pairs...")
    val_lined_paths, val_original_paths, val_num_samples = get_validation_pairs()

    # Get a sample image for progress visualization
    sample_image_path = lined_paths[0] if lined_paths else None
    target_image_path = original_paths[0] if original_paths else None

    # Get a sample ruled image for progress visualization (real ruled paper)
    ruled_image_path = None
    if os.path.exists(RULED_DIR):
        ruled_files = [f for f in os.listdir(RULED_DIR)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if ruled_files:
            ruled_image_path = os.path.join(RULED_DIR, ruled_files[0])
            print(f"Using ruled image for progress: {ruled_files[0]}")

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
    train_model(model, lined_paths, original_paths, num_samples,
                val_lined_paths, val_original_paths, val_num_samples,
                sample_image_path, target_image_path, ruled_image_path)

    print("\nTraining complete!")
    print(f"Model saved to: {MODEL_PATH}")


def process_ruled_images(model, limit=None, compare=False):
    """Process images from Ruled directory and save to results.

    These are real images with ruled lines (no corresponding unlined versions).
    Since there's no target/ground truth, comparisons show 2 panels only.

    Args:
        model: The trained model
        limit: If specified, only process this many images
        compare: If True, create 2-panel comparison images (input | output)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Get all image files
    image_files = sorted([
        f for f in os.listdir(RULED_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if limit and limit < len(image_files):
        # Randomly select images when limit is specified
        image_files = random.sample(image_files, limit)

    mode_str = "comparison" if compare else "output"
    print(f"Processing {len(image_files)} ruled images ({mode_str} mode)...")

    for i, filename in enumerate(image_files):
        input_path = os.path.join(RULED_DIR, filename)
        output_path = os.path.join(RESULTS_DIR, filename)

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
    print(f"Output saved to: {RESULTS_DIR}")


def inference(limit=None, compare=False):
    """Load trained model and process images."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first with: python line_remover.py --train")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)

    process_all_images(model, limit=limit, compare=compare)


def inference_ruled(limit=None, compare=False):
    """Load trained model and process ruled images."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first with: python line_remover.py --train")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)

    process_ruled_images(model, limit=limit, compare=compare)


def inference_file(file_path, compare=False):
    """Load trained model and process a single file."""
    # Resolve to absolute path
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first with: python line_remover.py --train")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)

    # Determine output path - save to results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = os.path.basename(file_path)
    output_path = os.path.join(RESULTS_DIR, filename)

    print(f"Processing: {file_path}")

    try:
        result_img = remove_lines(model, file_path)

        if compare:
            input_img = Image.open(file_path).convert('RGB')
            comparison = create_comparison_image(input_img, result_img)
            comparison.save(output_path, quality=95)
        else:
            result_img.save(output_path, quality=95)

        print(f"Output saved to: {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Remove ruled lines from sketch images')
    parser.add_argument('--train', action='store_true',
                        help='Train the model from scratch')
    parser.add_argument('--resume', action='store_true',
                        help='Continue training from saved model')
    parser.add_argument('--process', action='store_true',
                        help='Process images from lines_added to lines_removed')
    parser.add_argument('--ruled', action='store_true',
                        help='Process images from Ruled directory to results')
    parser.add_argument('--compare', action='store_true',
                        help='Create side-by-side comparison images (input | output)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to process')
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='Process a specific file (relative or absolute path)')
    args = parser.parse_args()

    if args.train or args.resume:
        train(resume=args.resume)
    elif args.file:
        inference_file(args.file, compare=args.compare)
    elif args.ruled:
        inference_ruled(limit=args.limit, compare=args.compare)
    elif args.process or args.compare:
        inference(limit=args.limit, compare=args.compare)
    else:
        print("Usage:")
        print("  python line_remover.py --train              Train the model from scratch")
        print("  python line_remover.py --resume             Continue training from saved model")
        print("  python line_remover.py --process            Process all images (lines_added)")
        print("  python line_remover.py --ruled              Process ruled images to results/")
        print("  python line_remover.py --ruled --compare    With side-by-side comparisons")
        print("  python line_remover.py --compare            Create side-by-side comparisons")
        print("  python line_remover.py --process --limit 5  Process 5 images")
        print("  python line_remover.py -f path/to/image.jpg Process a specific file")
        print("  python line_remover.py -f image.jpg --compare  With side-by-side comparison")


if __name__ == '__main__':
    main()

