The directory ./data/Sketches contains three subdirectoires.
'Unruled' contains a collection of sketches of boats on plain paper.
'Ruled' contains a similiar collection, but drawn on lined paper.
'lines_added' is used to store output files.

Notice that there is variation in the lines in the images. 
There are variations in darkness and color and angle and vertical spacing.
Some of the lines may be ragged or discontinuous.

I want to write a python progam that takes in images from the 'Unruled' directory and modifies them by adding lines that that resemble the lines found in images in the 'Ruled' directory. The new image should be saved in the 'lines_added' directory. The name of the new image should be the same as the original, with the addition of an integer suffix. This suffix should start at _00001 and increment by one for each new file. 

# step 2
write a new program in python called "line_remover" using tensorflow that performs the following functions:
1. Randomly select an image from ./data/Sketches/lines_added.
2. identify the corresponding image in ./data/Sketches/Unruled. This file will have the same file name, except it will not have the integer suffix such as "_00001"
3. Use a convolutional neural network similar to a denoising encoder/decoder. The input to this network will be the image from step one. The target will be the image from step 2. The goal is to make the result of the neural network match the image rfom step 2 as closely as possible.
4. Try to keep the total number of model parameters less than 10,000,000
5. if possible, use the GPU to accelerate model training
