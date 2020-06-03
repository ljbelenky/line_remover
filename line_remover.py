import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from PIL import Image
import os
from sklearn.decomposition import NMF

class LineRemover:
    def __init__(self, fname):
        self.fname = fname
        self.image = Image.open(fname).resize((400,500)) #width, height
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



    def determine_presumptive_background(self):
        '''
        Cluster by pixel color
        The cluster with larger average value is presumed to be background
        '''
        if 'presumptive_background' not in self.pixel_array:
            self.pixel_array['presumptive_background'] = KMeans(4).fit(self.pixel_array[['value']]).labels_

            groups = self.pixel_array[['value', 'presumptive_background']].groupby('presumptive_background').mean()['value']

            larger = groups.idxmax()

            self.pixel_array['presumptive_background'] = self.pixel_array['presumptive_background'] == larger


    def presumptive_background_image(self, background = True):
        
        if 'presumptive_background' not in self.pixel_array:
            self.determine_presumptive_background()

        orientation = self.width, self.height
        
        mask = self.pixel_array['presumptive_background'].values
        if not background: mask = (1-mask)

        array = mask.reshape(orientation) * self.array

        return Image.fromarray(array.astype('uint8'))

    def determine_row_average(self):
        if 'row_average' not in self.pixel_array:

            row_average = self.pixel_array.groupby('y').mean()['value'].reset_index().rename(columns = {'value':'row_average'})
            self._pixel_array = self.pixel_array.merge(row_average, on = 'y')
            self._pixel_array['delta_to_row_average'] = self.pixel_array['value'] - self.pixel_array['row_average']


    @property
    def data(self):
        self.determine_presumptive_background()
        self.determine_row_average()
        return self.pixel_array


    def plot_by_cluster(self):
        pass

            


    def __repr__(self):
        return f'LineRemover for file {self.fname}'



if __name__ == '__main__':

    ruled_dir = 'Sketches/Ruled'
    ruled_iamges = [f for f in os.listdir(ruled_dir) if os.path.isfile(os.path.join(ruled_dir, f))]


    for f in ruled_iamges:

        l1 = LineRemover(os.path.join(ruled_dir, f))
        break