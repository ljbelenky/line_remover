import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from PIL import Image
import os
from sklearn.decomposition import NMF
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler as SS

class LineRemover:
    def __init__(self, fname, width = 600):
        self.fname = fname
        image = Image.open(fname)
        
        original_width, original_height = image.size

        self.width = width
        self.height = int(width * original_height/original_width)
        
        self.image = image.resize((self.width, self.height))
        

    @property
    def array(self):
        if '_array' not in self.__dict__:
            self._array = np.array(self.image)
        return self._array

    
    @property
    def pixel_array(self):
        if '_pixel_array' not in self.__dict__:
            xs = []
            ys = []
                
            for j in range(self.height):
                for i in range(self.width):
                    xs.append(i)
                    ys.append(j)
            

            self._pixel_array = pd.DataFrame({'x':xs,
                                              'y':ys,
                                              'value':self.array.flatten()})

        return self._pixel_array

    def show(self):
        self.image.show()


    @property
    def padded_array(self):
        '''
        Creates a padded 2D array slightly larger than the orginal using the mode of the original as the broder
        '''
        if '_padded_array' not in self.__dict__:

            padding = (self.filters[0].shape[1]-1)//2
            pad_value = mode(self.array.flatten())[0]
            self._padded_array = (np.ones(shape = (2*padding + self.height, 2*padding+self.width)) * pad_value).astype(np.uint8)
            self._padded_array[padding:padding+self.height, padding:padding+self.width] = self.array

        return self._padded_array

    
    @property
    def filters(self):

        # TODO: Use PCA or clustering to develop filters from image

        if '_filters' not in self.__dict__:
            f0 = np.zeros((7,7))
            f0[3] = 1

            f1 = np.zeros((7,7))
            f1[[2,3,4],:] = 1

            f2 = np.zeros((7,7))
            f2[[1,2,3,4,5]] = 1

            f3 = np.eye(7)
            f4 = f3[::-1]

            filters = [f0, f1, f2]

            self._filters = filters + [1-f for f in filters] + [f.T for f in filters] + [(1-f).T for f in filters] + [f3, f4] +[1-f3, 1-f4]
        return self._filters

    def calcualte_subimages(self):

        if 'subimage' not in self.pixel_array:
            print('calculating sub images')
            filter_size = self.filters[0].shape[1]
            padded_array = self.padded_array

            def calculate_sub_image(row):
                x, y = row.x, row.y
                return padded_array[y:y+filter_size, x:x+filter_size]

            self.pixel_array['subimage'] = self.pixel_array.apply(calculate_sub_image, axis = 1)


    def convolve(self):
        if 'subimage' not in self.pixel_array:
            self.calcualte_subimages()

        if 'f0' not in self.pixel_array:

            def _convolve(row, filter):
                a = row['subimage'].flatten()
                b = filter.flatten()
                return a@b/49   #divide by number of pixels in filter

            for i,f in enumerate(self.filters):
                self.pixel_array[f'f{i}'] = self.pixel_array.apply(_convolve, axis = 1, args = (f,))

        return self.pixel_array


    def cluster(self,k=8):
        print(f'Clustering with k={k}')
        X = SS().fit_transform(self.clustering_data)
        self.pixel_array['label'] = KMeans(k).fit(X).labels_
            


    def plot_clusters(self, overlay = True):
        if 'label' not in self.pixel_array:
            self.cluster()
        
        if overlay:
            c = self.pixel_array['label'].values.reshape((self.height, -1))
            plt.matshow(c)
            plt.show()

        # else:
        #     clusters = self.pixel_array['label'].max()
        #     fig, axs = plt.subplots(self.pixel_array['label'].max() - 1)

        #     for 



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

        
        mask = self.pixel_array['presumptive_background'].values.reshape(self.height, self.width)
        if not background: mask = (1-mask)

        array = mask * self.array

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
        self.convolve()
        return self.pixel_array


    @property 
    def clustering_data(self):
        excluded_columns = ['x','y','subimage','presumptive_background']
        return self.data[[col for col in self.data.columns if col not in excluded_columns ]]

            


    def __repr__(self):
        return f'LineRemover for file {self.fname}'



if __name__ == '__main__':

    ruled_dir = 'Sketches/Ruled'
    ruled_images = [f for f in os.listdir(ruled_dir) if os.path.isfile(os.path.join(ruled_dir, f))]


    for i, f in enumerate(ruled_images):

        if i < 2: continue

        l1 = LineRemover(os.path.join(ruled_dir, f))
        break