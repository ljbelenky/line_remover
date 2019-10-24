# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import numpy as np
from skimage import color
import matplotlib.pyplot as plt


class RulerGenerator:
    def __init__(self, shape, line_width, spacing, raggedness, color, color_variation, angle):
        # inner_width = int(line_width/3)
        # outer_width = min(int(line_width), 1)
        h, w = shape

        basis = np.array([0]*spacing + [color]*line_width)

        lines_array = np.array(
            np.tile(basis, int((h*w)/len(basis)))).reshape(h, w)

        return lines_array

        # lines_array = np.zeros(shape=shape)

        # number_of_lines = shape[0]/(spacing+line_width))


class SketchDataGenerator():
    '''SketchDataGenerator yields pairs of sketches with/without ruled liens and image augmentation'''

    def __init__(self, flip_horizontal=False, flip_vertical=False, h_shift=0, v_shift=0, rotate=0, dilate=0, contrast=0):
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.h_shift = h_shift
        self.v_shift = v_shift
        self.rotate = rotate
        self.dilate = dilate
        self.contrast = contrast

    def flow_from_directory(self, directory):

        filenames = os.listdir(directory)
        while True:
            for filename in filenames:
                sketch = Image.open(os.path.join(directory, filename))
                # sketch = np.asarray(sketch)
                yield sketch


if __name__ == '__main__':
    sdg = SketchDataGenerator()

    sketches = sdg.flow_from_directory('Sketches/Unruled')

    for sketch in sketches:
        break
