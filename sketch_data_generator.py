# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import numpy as np
from skimage import color
import matplotlib.pyplot as plt


class RulerGenerator:
    def __init__(self, shape, line_width, spacing, v_offset, raggedness, color, color_variation, angle):
        # inner_width = int(line_width/3)
        # outer_width = min(int(line_width), 1)
        h, w = shape

        basis = np.array([0]*spacing + [1]*line_width).astype('uint8')
        column = np.tile(basis, 1+int(2*h/len(basis)))[0:2*h]
        lines_array = np.tile(column, 2*w).reshape(2*w, 2*h).T

        self.color_array = np.random.random(size = shape)*color_variation + color

        self.image = Image.fromarray(lines_array)
        self.image = self.image.rotate(angle)

        left = int(w/2)
        right = left + w
        top = int(h/2 + v_offset*len(basis))
        bottom = top + h

        self.image = self.image.crop((left, top, right, bottom))



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
    rg = RulerGenerator((400,320), 3, 25, 3, 50, 128, 10, 10)
    i = rg.image
    i.show()