#!/usr/bin/env python3
"""
GAN Line Remover - A pix2pix style GAN to remove ruled lines from sketch images.

Uses:
- Generator: U-Net architecture
- Discriminator: PatchGAN (judges 70x70 patches)
- Loss: Adversarial + L1 reconstruction
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
RULED_DIR = os.path.join(DATA_DIR, 'Ruled')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_gan')
PROGRESS_DIR = os.path.join(BASE_DIR, 'progress_gan')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'gan_checkpoints')
GENERATOR_PATH = os.path.join(BASE_DIR, 'gan_generator.keras')

# Model parameters
IMG_HEIGHT = 512
IMG_WIDTH = 512
BATCH_SIZE = 1
EPOCHS = 200
SAMPLES_PER_EPOCH = 500  # Random subset of images per epoch
LAMBDA_L1 = 100  # Weight for L1 loss vs adversarial loss


def get_base_name(filename):
    """Extract base name from a filename by removing the suffix like _00001."""
    name_without_ext = os.path.splitext(filename)[0]
    match = re.match(r'(.+)_\d{5}$', name_without_ext)
    if match:
        return match.group(1)
    return name_without_ext


def find_original_image(lined_filename):
    """Find the corresponding original image in Unruled directory."""
    base_name = get_base_name(lined_filename)
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
    # Normalize to [-1, 1] for GAN training
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img


def load_image_pair(lined_path, original_path):
    """Load a pair of images (lined and original)."""
    lined_img = load_image_tf(lined_path)
    original_img = load_image_tf(original_path)
    return lined_img, original_img


def get_image_pairs():
    """Get all valid image path pairs from lines_added and Unruled directories."""
    lined_files = [f for f in os.listdir(LINES_ADDED_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(lined_files)} images in lines_added directory")

    lined_paths = []
    original_paths = []

    for lined_file in lined_files:
        lined_path = os.path.join(LINES_ADDED_DIR, lined_file)
        original_path = find_original_image(lined_file)

        if original_path is None:
            continue

        lined_paths.append(lined_path)
        original_paths.append(original_path)

    num_samples = len(lined_paths)
    print(f"Found {num_samples} valid image pairs")

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


def sample_random_subset(lined_paths, original_paths, n_samples):
    """Randomly sample n_samples image pairs."""
    indices = np.random.permutation(len(lined_paths))[:n_samples]
    sampled_lined = [lined_paths[i] for i in indices]
    sampled_original = [original_paths[i] for i in indices]
    return sampled_lined, sampled_original


def downsample(filters, size, apply_batchnorm=True):
    """Downsampling block for encoder."""
    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    """Upsampling block for decoder."""
    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                       kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result


def build_generator():
    """
    Build U-Net generator.

    512x512 input -> series of downsamples -> bottleneck -> upsamples with skip connections -> 512x512 output
    """
    inputs = keras.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # 256x256
        downsample(128, 4),   # 128x128
        downsample(256, 4),   # 64x64
        downsample(512, 4),   # 32x32
        downsample(512, 4),   # 16x16
        downsample(512, 4),   # 8x8
        downsample(512, 4),   # 4x4
        downsample(512, 4),   # 2x2
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),   # 4x4
        upsample(512, 4, apply_dropout=True),   # 8x8
        upsample(512, 4, apply_dropout=True),   # 16x16
        upsample(512, 4),   # 32x32
        upsample(256, 4),   # 64x64
        upsample(128, 4),   # 128x128
        upsample(64, 4),    # 256x256
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                   kernel_initializer=initializer)  # 512x512

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)
    # Explicit float32 output for mixed precision compatibility
    x = layers.Activation('tanh', dtype='float32')(x)

    return keras.Model(inputs=inputs, outputs=x, name='generator')


def build_discriminator():
    """
    Build PatchGAN discriminator.

    Takes both input image and target/generated image.
    Outputs a 30x30 patch of predictions.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = keras.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='input_image')
    tar = keras.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='target_image')

    x = layers.concatenate([inp, tar])  # 512x512x6

    down1 = downsample(64, 4, False)(x)   # 256x256
    down2 = downsample(128, 4)(down1)     # 128x128
    down3 = downsample(256, 4)(down2)     # 64x64

    zero_pad1 = layers.ZeroPadding2D()(down3)  # 66x66
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1)  # 63x63

    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # 65x65
    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2)  # 62x62
    # Explicit float32 output for mixed precision compatibility
    last = layers.Activation('linear', dtype='float32')(last)

    return keras.Model(inputs=[inp, tar], outputs=last, name='discriminator')


class Pix2Pix:
    """Pix2Pix GAN trainer."""

    def __init__(self):
        self.generator = build_generator()
        self.discriminator = build_discriminator()

        self.generator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.loss_object = keras.losses.BinaryCrossentropy(from_logits=True)

        # For progress tracking
        self.best_gen_loss = float('inf')
        os.makedirs(PROGRESS_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        # Sample image for progress visualization
        self.sample_input = None
        self.sample_target = None

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        return real_loss + generated_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (LAMBDA_L1 * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
                disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                 self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                      self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                      self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                          self.discriminator.trainable_variables))

        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

    def save_progress_image(self, epoch, gen_loss):
        """Save a comparison image showing progress."""
        if self.sample_input is None:
            return

        # Generate output
        prediction = self.generator(self.sample_input, training=False)

        # Convert from [-1, 1] to [0, 255]
        input_img = ((self.sample_input[0].numpy() + 1) * 127.5).astype(np.uint8)
        output_img = ((prediction[0].numpy() + 1) * 127.5).astype(np.uint8)

        # Create comparison
        input_pil = Image.fromarray(input_img)
        output_pil = Image.fromarray(output_img)

        width, height = input_pil.size
        comparison = Image.new('RGB', (width * 2, height))
        comparison.paste(input_pil, (0, 0))
        comparison.paste(output_pil, (width, 0))

        filename = f"{gen_loss:.6f}_epoch{epoch:04d}.jpg"
        output_path = os.path.join(PROGRESS_DIR, filename)
        comparison.save(output_path, quality=95)
        print(f" -> Saved progress: {filename}")

    def train(self, lined_paths, original_paths, num_samples, epochs=EPOCHS):
        """Train the GAN with random sampling each epoch."""
        samples_per_epoch = min(SAMPLES_PER_EPOCH, num_samples)
        steps_per_epoch = max(1, samples_per_epoch // BATCH_SIZE)

        print(f"\nStarting training for {epochs} epochs")
        print(f"Using {samples_per_epoch} random samples per epoch ({steps_per_epoch} steps)")
        print(f"Total available images: {num_samples}")
        print(f"Generator params: {self.generator.count_params():,}")
        print(f"Discriminator params: {self.discriminator.count_params():,}")

        for epoch in range(epochs):
            start = time.time()

            # Sample a random subset for this epoch
            sampled_lined, sampled_original = sample_random_subset(
                lined_paths, original_paths, samples_per_epoch)
            dataset = create_dataset_from_paths(sampled_lined, sampled_original)

            # Get sample for visualization (first epoch only)
            if self.sample_input is None:
                for input_image, target in dataset.take(1):
                    self.sample_input = input_image
                    self.sample_target = target

            gen_loss_sum = 0
            disc_loss_sum = 0
            step_count = 0

            for input_image, target in dataset:
                gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(
                    input_image, target)

                gen_loss_sum += gen_total_loss.numpy()
                disc_loss_sum += disc_loss.numpy()
                step_count += 1

            avg_gen_loss = gen_loss_sum / max(step_count, 1)
            avg_disc_loss = disc_loss_sum / max(step_count, 1)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f} - "
                  f"{time.time()-start:.1f}s")

            # Save progress if generator improved
            if avg_gen_loss < self.best_gen_loss:
                self.best_gen_loss = avg_gen_loss
                self.save_progress_image(epoch + 1, avg_gen_loss)
                self.generator.save(GENERATOR_PATH)
                print(f" -> Model saved")

            # Checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"gen_epoch{epoch+1:04d}.keras")
                self.generator.save(checkpoint_path)

        print("\nTraining complete!")
        print(f"Generator saved to: {GENERATOR_PATH}")


def remove_lines(generator, image_path, output_path=None):
    """Use the trained generator to remove lines from an image."""
    original_img = Image.open(image_path).convert('RGB')
    original_size = original_img.size

    img = original_img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0  # Normalize to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)

    result = generator(img_array, training=False)[0].numpy()

    # Convert back to [0, 255]
    result = ((result + 1) * 127.5).astype(np.uint8)
    result_img = Image.fromarray(result)
    result_img = result_img.resize(original_size, Image.Resampling.LANCZOS)

    if output_path:
        result_img.save(output_path, quality=95)

    return result_img


def create_comparison_image(input_img, output_img):
    """Create a side-by-side comparison image."""
    width, height = input_img.size
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(input_img, (0, 0))
    comparison.paste(output_img, (width, 0))
    return comparison


def process_ruled_images(generator, limit=None, compare=False):
    """Process images from Ruled directory and save to results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(RULED_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if limit:
        image_files = image_files[:limit]

    mode_str = "comparison" if compare else "output"
    print(f"Processing {len(image_files)} ruled images ({mode_str} mode)...")

    for i, filename in enumerate(image_files):
        input_path = os.path.join(RULED_DIR, filename)
        output_path = os.path.join(RESULTS_DIR, filename)

        try:
            result_img = remove_lines(generator, input_path)

            if compare:
                input_img = Image.open(input_path).convert('RGB')
                comparison = create_comparison_image(input_img, result_img)
                comparison.save(output_path, quality=95)
            else:
                result_img.save(output_path, quality=95)

            print(f"[{i+1}/{len(image_files)}] Processed: {filename}")
        except Exception as e:
            print(f"[{i+1}/{len(image_files)}] Error processing {filename}: {e}")

    print(f"\nDone! Processed {len(image_files)} images.")
    print(f"Output saved to: {RESULTS_DIR}")


def train(resume=False):
    """Train the GAN."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) available: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
        print("Mixed precision (bfloat16) enabled")
    else:
        print("No GPU available, using CPU")

    print("\nScanning for image pairs...")
    lined_paths, original_paths, num_samples = get_image_pairs()

    if num_samples == 0:
        print("Error: No valid image pairs found.")
        return

    gan = Pix2Pix()

    if resume and os.path.exists(GENERATOR_PATH):
        print(f"\nLoading generator from {GENERATOR_PATH}...")
        gan.generator = keras.models.load_model(GENERATOR_PATH)

    gan.generator.summary()
    gan.discriminator.summary()

    gan.train(lined_paths, original_paths, num_samples)


def inference_ruled(limit=None, compare=False):
    """Load trained generator and process ruled images."""
    if not os.path.exists(GENERATOR_PATH):
        print(f"Error: Generator not found at {GENERATOR_PATH}")
        print("Please train the model first with: python gan_line_remover.py --train")
        return

    print(f"Loading generator from {GENERATOR_PATH}...")
    generator = keras.models.load_model(GENERATOR_PATH)

    process_ruled_images(generator, limit=limit, compare=compare)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='GAN-based line removal from sketch images')
    parser.add_argument('--train', action='store_true',
                        help='Train the GAN from scratch')
    parser.add_argument('--resume', action='store_true',
                        help='Continue training from saved generator')
    parser.add_argument('--ruled', action='store_true',
                        help='Process images from Ruled directory to results_gan/')
    parser.add_argument('--compare', action='store_true',
                        help='Create side-by-side comparison images')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to process')
    args = parser.parse_args()

    if args.train or args.resume:
        train(resume=args.resume)
    elif args.ruled:
        inference_ruled(limit=args.limit, compare=args.compare)
    else:
        print("Usage:")
        print("  python gan_line_remover.py --train           Train the GAN from scratch")
        print("  python gan_line_remover.py --resume          Continue training")
        print("  python gan_line_remover.py --ruled           Process ruled images")
        print("  python gan_line_remover.py --ruled --compare With side-by-side comparisons")


if __name__ == '__main__':
    main()
