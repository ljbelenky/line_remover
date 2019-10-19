# line_remover

A project using CNNs to remove ruled lines from sketches

## Raw Data

There are 1124 sketches in the 'Sketches' directory. (Some files appear to be corrupt, will need to ivnestigate this.)

The histograms below show the heights, widths and aspect ratios of these images.

![Horizontal](readme_images/width.jpg)
![Vertical](readme_images/height.jpg)
![Aspect](readme_images/aspect.jpg)

Based on these measurements, all images will be resized to a standard size of height = 4000, width = 3200 pixels, or some multiple of these dimensions (i.e., 400 x 320, 800 x 640, etc)

This will be done by first determining the aspect ratio for each image. If the aspect ratio is greater than 1.25 (height/width), the image will be resized to 4000 pixels high and something less than 3200 pixels wide. It will then be padded on the sides to fill to 3200 pixel width.

If the aspect ratio is less than 1.25, the image will be resized to 3200 pixels wide and the corresponding height. It will then be padded top and bottom to a height of 4000 pixels.

## Image Color Format

Because the sketches are in charcoal and essentailly grayscale, I thought it would make sense to use the LAB image format. In this format, the first channel, Lightness, encodes the most signficant information about the sketch. The remaining two channels provide some color shading information.
