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
    size_dictionary = {'small': 200, 'medium': 500, 'large': 4000}

    for size_name, new_height in size_dictionary.items():

        new_width = int(new_height/1.25)

        for source in ['Ruled', 'Unruled']:
            source_dir = os.path.join('Sketches', source)
            destination_dir = os.path.join('LAB_Arrays', size_name, source)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)

            for filename in os.listdir(source_dir):
                try:
                    sketch = Image.open(os.path.join(source_dir, filename))
                    width, height = sketch.size
                    aspect = height/width

                    if aspect >= 1.25:
                        size = (int(new_height/aspect), new_height)
                    else:
                        size = (new_width, int(new_width*aspect))
                    sketch = sketch.resize(size=size)
                    sketch = sketch.convert(mode='RGB')
                    LAB_array = color.rgb2lab(sketch)

                    median_L = np.median(LAB_array[:, :, 0])
                    median_A = np.median(LAB_array[:, :, 1])
                    median_B = np.median(LAB_array[:, :, 2])

                    new_L = np.ones(shape=(new_height, new_width)) * median_L
                    new_A = np.ones(shape=(new_height, new_width)) * median_A
                    new_B = np.ones(shape=(new_height, new_width)) * median_B

                    width, height = sketch.size

                    left = int((new_width - width) / 2)
                    right = left + width

                    bottom = int((new_height - height)/2)
                    top = bottom + height

                    new_L[bottom:top, left:right] = LAB_array[:, :, 0]
                    new_A[bottom:top, left:right] = LAB_array[:, :, 1]
                    new_B[bottom:top, left:right] = LAB_array[:, :, 2]

                    new_LAB_array = np.moveaxis(
                        np.array([new_L, new_A, new_B]), 0, 2)

                    new_filename = os.path.splitext(filename)[0]
                    new_filename = os.path.join(destination_dir, new_filename)
                    np.save(new_filename+'.npy', new_LAB_array)

                    print(source, ':', size_name, ':', filename)
                except:
                    print('!!!!!!!!!!!!!!!!Error on ', filename)

                # new_image = color.lab2rgb(new_LAB_array)
                # new_image *= 255
                # new_image = new_image.astype('uint8')
                # new_image = Image.fromarray(new_image)

                # return sketch, new_LAB_array  # , new_image
