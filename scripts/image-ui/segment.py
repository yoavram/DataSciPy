import io
import imageio
import numpy as np
import skimage.filters
import click
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def open_image(filename):
    return imageio.imread(filename)

def save_image(filename, image):
    imageio.imsave(filename, image)

def segment_image(image):
    grayscale = image.mean(axis=2) # convert to grayscale
    th = skimage.filters.threshold_otsu(grayscale) # find segmentation threshold
    segmented = grayscale > th # segment
    return segmented.astype(np.uint8) * 255 # convert to int

@click.command()
@click.argument("src", type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument("dst", type=click.Path(writable=True, dir_okay=False))
def main(src, dst):
    image = open_image(src)
    segmented = segment_image(image)
    save_image(dst, segmented)


if __name__ == '__main__':
    main()
