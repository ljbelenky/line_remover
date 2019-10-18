# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import numpy as np
from skimage import color
import matplotlib.pyplot as plt


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


def convert_all_to_lab():
    source_dirs = ['Sketches/Ruled', 'Sketches/Unruled']
    dest_dirs = ['LAB/Ruled', 'LAB/Unruled']

    h_size = []
    v_size = []

    for source, dest in zip(source_dirs, dest_dirs):
        for filename in os.listdir(source):
            try:
                sketch = Image.open(os.path.join(source, filename))
                h_size.append(sketch.size[0])
                v_size.append(sketch.size[1])
            except:
                pass
            # sketch = sketch.convert(mode='RGB')
            # sketch = color.rgb2lab(sketch)
            # sketch = Image.fromarray(sketch, mode='LAB')
            # return sketch
    return h_size, v_size
