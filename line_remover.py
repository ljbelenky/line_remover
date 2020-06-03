import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from PIL import Image
import os

class LineRemover:
    def __init__(self, fname):
        self.fname = fname
        self.image = Image.open(fname)
        self._pixel_array = None
        self._array = None

    @property
    def array(self):
        if self._array is None:
            self._array = np.array(self.image)
        return self._array

    @property
    def width(self):
        return self.array.shape[0]

    @property
    def height(self):
        return self.array.shape[1]

    @property
    def pixel_array(self):
        if self._pixel_array is None:
            xs = []
            ys = []
            for i in range(self.width):
                for j in range(self.height):
                    xs.append(i)
                    ys.append(j)
            

            self._pixel_array = pd.DataFrame({'x':xs,
                                              'y':ys,
                                              'value':self.array.flatten()})

        return self._pixel_array

    def show(self):
        self.image.show()



    def determine_presumtive_backgrounds(self):
        '''
        Cluster by pixel color
        The cluster with lightest average value is presumed to be background
        '''
        if 'presumtive_background' not in self.pixel_array:
            self.pixel_array['presumtive_background'] = KMeans(2).fit(self.pixel_array[['value']]).labels_
            


    def __repr__(self):
        return f'LineRemover for file {self.fname}'



if __name__ == '__main__':

    ruled_dir = 'Sketches/Ruled'
    ruled_iamges = [f for f in os.listdir(ruled_dir) if os.path.isfile(os.path.join(ruled_dir, f))]


    for f in ruled_iamges:

        l1 = LineRemover(os.path.join(ruled_dir, f))
        break