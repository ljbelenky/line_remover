
import numpy as np
from PIL import Image


def merge_images(im1, im2):
    w = im1.size[0] + im2.size[0]
    h = max(im1.size[1], im2.size[1])
    im = Image.new("RGBA", (w, h))

    im.paste(im1)
    im.paste(im2, (im1.size[0], 0))

    return im


class RulerGenerator:
    # Line color presets that match real ruled paper
    # Note: Lower values = darker lines when blended
    COLOR_PRESETS = {
        'blue': {'r': 180, 'g': 200, 'b': 255},      # Light blue
        'cyan': {'r': 100, 'g': 220, 'b': 220},      # Cyan/turquoise (like real ruled paper)
        'dark_blue': {'r': 150, 'g': 170, 'b': 230}, # Darker blue
        'gray': {'r': 160, 'g': 160, 'b': 160},      # Gray lines
        'light_gray': {'r': 190, 'g': 190, 'b': 190}, # Light gray
        'green': {'r': 180, 'g': 220, 'b': 180},     # Green tinted (some paper)
        # Faint presets matching real scanned ruled paper
        'faint_gray': {'r': 220, 'g': 220, 'b': 220},      # Very faint gray (like real paper)
        'faint_blue': {'r': 220, 'g': 225, 'b': 240},      # Very faint blue tint
        'very_faint': {'r': 235, 'g': 235, 'b': 235},      # Barely visible gray
    }

    MARGIN_COLORS = {
        'red': {'r': 255, 'g': 180, 'b': 180},       # Red margin line
        'pink': {'r': 255, 'g': 200, 'b': 200},      # Pink margin line
    }

    def __init__(self, shape, line_width, lines, v_offset, raggedness, color,
                 color_variation, angle, line_color=None, add_margin=None,
                 waviness=0.0):
        """
        Generate ruled paper lines.

        Args:
            shape: (width, height) tuple
            line_width: Width of lines in pixels
            lines: Number of horizontal lines
            v_offset: Vertical offset (0-1)
            raggedness: Line texture roughness (0-1)
            color: Base intensity (0-255, higher = darker lines)
            color_variation: Random variation in line darkness
            angle: Rotation angle in degrees
            line_color: Color preset name ('blue', 'gray', etc.) or None for random
            add_margin: Add vertical margin line ('red', 'pink', or None/False)
            waviness: Amount of horizontal waviness (0-1, 0=straight, 1=very wavy)
        """
        lines = lines or 12

        w, h = shape
        spacing = h // lines
        basis = np.array([0] * spacing + [1] * line_width).astype('uint8')
        column = np.tile(basis, 1 + int(2 * h / len(basis)))[0:2 * h]
        lines_array = np.tile(column, 2 * w).reshape(2 * w, 2 * h).T

        # Apply waviness - shift each column horizontally based on sine wave
        if waviness > 0:
            lines_array = self._apply_waviness(lines_array, waviness, w, h)

        # Color variation per pixel
        color_array = np.random.normal(size=(2 * h, 2 * w)) * color_variation + color
        color_array = np.clip(color_array, 0, 255)
        self.color_array = color_array

        # Raggedness - random gaps in lines
        ragged_array = (np.random.random(size=(2 * h, 2 * w)) > raggedness).astype(np.float32)

        # Combine intensity (use float to avoid overflow)
        lines_array = lines_array.astype(np.float32) * color_array * ragged_array

        # Select line color
        if line_color is None:
            # Random color with weighted probabilities (blue most common)
            color_choices = ['blue', 'blue', 'blue', 'dark_blue', 'gray', 'light_gray']
            line_color = np.random.choice(color_choices)

        line_rgb = self.COLOR_PRESETS.get(line_color, self.COLOR_PRESETS['blue'])

        # Create RGB channels - lines are darker versions of the line color
        # Where lines_array is high, we darken the pixel toward the line color
        red = np.where(lines_array > 0,
                       255 - (lines_array * (255 - line_rgb['r']) / 255),
                       255).astype('uint8')
        green = np.where(lines_array > 0,
                         255 - (lines_array * (255 - line_rgb['g']) / 255),
                         255).astype('uint8')
        blue = np.where(lines_array > 0,
                        255 - (lines_array * (255 - line_rgb['b']) / 255),
                        255).astype('uint8')

        RGB_array = np.stack([red, green, blue], axis=2)

        # Add margin line if requested
        if add_margin is None:
            # 30% chance of adding a margin line
            add_margin = np.random.random() < 0.3
            if add_margin:
                add_margin = np.random.choice(['red', 'pink'])

        if add_margin:
            RGB_array = self._add_margin_line(RGB_array, add_margin, w, h)

        image = Image.fromarray(RGB_array, mode='RGB')
        image = image.rotate(angle)

        left = int(w / 2)
        right = left + w
        top = int(h / 2 + v_offset * len(basis))
        bottom = top + h

        self.image = image.crop((left, top, right, bottom))

    def _apply_waviness(self, lines_array, waviness, w, h):
        """Apply horizontal waviness to lines using sine waves."""
        # Create a shifted version of the array
        result = np.zeros_like(lines_array)

        # Random parameters for the wave
        frequency = np.random.uniform(0.5, 2.0)  # How many waves across the width
        amplitude = waviness * 3  # Max shift in pixels
        phase = np.random.uniform(0, 2 * np.pi)

        for col in range(2 * w):
            # Calculate shift for this column
            shift = int(amplitude * np.sin(2 * np.pi * frequency * col / (2 * w) + phase))
            # Roll the column
            result[:, col] = np.roll(lines_array[:, col], shift)

        return result

    def _add_margin_line(self, RGB_array, margin_color, w, h):
        """Add a vertical margin line on the left side."""
        margin_rgb = self.MARGIN_COLORS.get(margin_color, self.MARGIN_COLORS['red'])

        # Margin line position (8-12% from left edge of the VISIBLE cropped area)
        # The visible area starts at x = w/2, so we add that offset
        left_edge = int(w / 2)
        margin_pos = left_edge + int(w * np.random.uniform(0.08, 0.12))
        margin_width = np.random.randint(1, 3)  # 1-2 pixels wide

        # Add some variation to the line
        for x in range(margin_pos, min(margin_pos + margin_width, 2 * w)):
            for y in range(2 * h):
                # Some raggedness
                if np.random.random() > 0.05:
                    # Blend with existing color
                    intensity = np.random.uniform(0.6, 0.9)
                    RGB_array[y, x, 0] = int(RGB_array[y, x, 0] * (1 - intensity) + margin_rgb['r'] * intensity)
                    RGB_array[y, x, 1] = int(RGB_array[y, x, 1] * (1 - intensity) + margin_rgb['g'] * intensity)
                    RGB_array[y, x, 2] = int(RGB_array[y, x, 2] * (1 - intensity) + margin_rgb['b'] * intensity)

        return RGB_array


if __name__ == '__main__':
    # Test with different color presets
    kwargs = {
        'shape': (500, 500),
        'line_width': np.clip(np.random.randint(-2, 4), 1, 4),
        'lines': np.random.randint(15, 40),
        'v_offset': np.random.random(),
        'raggedness': -0.15 + np.random.random() / 2,
        'color': np.random.randint(60, 120),  # Lighter lines
        'color_variation': np.random.randint(-15, 16),
        'angle': np.random.randint(-7, 8),
        'line_color': None,  # Random color
        'add_margin': None,  # Random margin
        'waviness': np.random.uniform(0, 0.3),
    }

    print(f"Generating with: line_color={kwargs.get('line_color')}, waviness={kwargs.get('waviness'):.2f}")
    rg = RulerGenerator(**kwargs)
    rg.image.show()
