
import numpy as np
from PIL import Image

def merge_images(im1, im2):
    w = im1.size[0] + im2.size[0]
    h = max(im1.size[1], im2.size[1])
    im = Image.new("RGBA", (w, h))

    im.paste(im1)
    im.paste(im2, (im1.size[0], 0))

    return im


class RulerGenerator:
    def __init__(self, shape, line_width, lines, v_offset, raggedness, color, color_variation, angle):

        lines = lines or 12
        # assert 0 <= raggedness <= 1
        # assert 0 <= color <= 255
        # assert 0 <= color_variation <= 255
        # assert -180 <= angle <= 180
        # assert 0 <= v_offset <= 1

        w, h = shape
        spacing = h//lines
        basis = np.array([0]*spacing + [1]*line_width).astype('uint8')
        column = np.tile(basis, 1+int(2*h/len(basis)))[0:2*h]
        lines_array = np.tile(column, 2*w).reshape(2*w, 2*h).T

        color_array = np.random.normal(
            size=(2*h, 2*w))*color_variation + color
        color_array = np.clip(color_array.astype('uint8'), 0, 255)
        self.color_array = color_array

        ragged_array = 1*(np.random.random(size=(2*h, 2*w))
                          > raggedness).astype('uint8')

        lines_array *= color_array
        lines_array *= ragged_array
        l = 255 - lines_array
        # color_offsets = np.random.randint(-60,60, 3)
        # red = np.clip(np.where(l==255,255, l/3+color_offsets[0]), 0,255).astype('uint8')
        # green = np.clip(np.where(l==255,255,l/3+color_offsets[1]),0,255).astype('uint8')
        # blue = np.clip(np.where(l==255,255, l/3+color_offsets[2]),0,255).astype('uint8')
        
        # RGB_array = np.stack([red,green,blue], axis =2)
        RGB_array = np.stack([l]*3, axis=2)
        image = Image.fromarray(RGB_array, mode = 'RGB')
        
        image = image.rotate(angle)

        left = int(w/2)
        right = left + w
        top = int(h/2 + v_offset*len(basis))
        bottom = top + h

        self.image = image.crop((left, top, right, bottom))


if __name__=='__main__':

    kwargs = {'shape': (500,500),
  'line_width': np.clip(np.random.randint(-2,3),1,4),
  'lines': np.random.randint(15,40),
  'v_offset': np.random.random(),
  'raggedness': -.15 + np.random.random()/2,
  'color': np.random.randint(100, 175),
  'color_variation': np.random.randint(-15, 16),
  'angle': np.random.randint(-7, 8)}
    rg = RulerGenerator(**kwargs)

    rg = RulerGenerator(**kwargs)
    
    rg.image.show()