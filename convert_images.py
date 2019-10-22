# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import numpy as np
from skimage import color
import matplotlib.pyplot as plt


if __name__ == '__main__':
    size_dictionary = {'small': 200, 'medium': 500, 'large': 4000}

    for size_name, new_height in size_dictionary.items():

        new_width = int(new_height/1.25)

        for source in ['Ruled', 'Unruled']:
            source_dir = os.path.join('Sketches', source)
            destination_dir = os.path.join(
                'grayscale_images', size_name, source)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)

            for filename in os.listdir(source_dir):
                try:
                    # while True:
                    sketch = Image.open(os.path.join(source_dir, filename))
                    width, height = sketch.size
                    aspect = height/width

                    if aspect >= 1.25:
                        size = (int(new_height/aspect), new_height)
                    else:
                        size = (new_width, int(new_width*aspect))
                    sketch = sketch.resize(size=size)
                    sketch = sketch.convert(mode='L')

                    L_array = np.asarray(sketch)
                    background = np.median(L_array).astype('uint8')

                    new_L = np.ones(shape=(new_height, new_width))*background
                    new_L = new_L.astype('uint8')

                    width, height = sketch.size

                    left = int((new_width - width) / 2)
                    right = left + width

                    bottom = int((new_height - height)/2)
                    top = bottom + height

                    new_L[bottom:top, left:right] = L_array

                    new_image = Image.fromarray(new_L)

                    new_filename = os.path.join(destination_dir, filename)
                    print(new_filename)
                    new_image.save(new_filename)

                except:
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!', filename)
