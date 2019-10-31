# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import numpy as np
from skimage import color
import matplotlib.pyplot as plt


class RulerGenerator:
    def __init__(self, shape, line_width, spacing, v_offset, raggedness, color, color_variation, angle):
        assert 0 <= raggedness <= 1
        assert 0 <= color <= 255
        assert 0 <= color_variation <= 255
        assert -180 <= angle <= 180
        assert 0 <= v_offset <= 1

        h, w = shape

        basis = np.array([0]*spacing + [1]*line_width).astype('uint8')
        column = np.tile(basis, 1+int(2*h/len(basis)))[0:2*h]
        lines_array = np.tile(column, 2*w).reshape(2*w, 2*h).T

        complementary_color = 255 - color

        color_array = np.random.normal(size = (2*h, 2*w))*color_variation + complementary_color - color_variation/2 
        color_array = color_array.astype('uint8')

        ragged_array = 1*(np.random.random(size = (2*h, 2*w))>raggedness).astype('uint8')

        lines_array *= color_array
        lines_array *= ragged_array

        image = Image.fromarray(lines_array)
        image = image.rotate(angle)

        left = int(w/2)
        right = left + w
        top = int(h/2 + v_offset*len(basis))
        bottom = top + h

        self.image = image.crop((left, top, right, bottom))

    @property
    def asarray(self):
        return np.asarray(self.image)


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

    def flow_from_directory(self, directory, result_type = 'arrays'):

        filenames = os.listdir(directory)
        while True:
            for filename in filenames:
                sketch = Image.open(os.path.join(directory, filename))
                lines = RulerGenerator(shape = (500,400), line_width = 3, spacing = 25, v_offset = .3, raggedness= .10, color =  128, color_variation= 25, angle = 10).asarray

                sketch_array = np.asarray(sketch)
                X = sketch_array - lines
                Y = sketch_array

                if result_type == 'arrays':
                    yield X, Y

                elif result_type == 'images':
                    yield Image.fromarray(np.hstack([X,Y]))

                else:
                    raise Exception


if __name__ == '__main__':
    rg = RulerGenerator(shape = (500,400), line_width = 3, spacing = 25, v_offset = .3, raggedness= .10, color =  128, color_variation= 25, angle = 10)
    i = rg.image
    # i.show()

    sdg = SketchDataGenerator()

    for sketch in sdg.flow_from_directory('grayscale_images/medium/Unruled', result_type='images'):
        sketch.show()