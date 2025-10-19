import io

import numpy as np
import skimage.filters
import scipy.misc
import click

def open_image(filename, mode="L"):
    return scipy.misc.imread(filename, mode=mode)

def save_image(filename, image):
    scipy.misc.imsave(filename, image)
    
def segment_image(image):
    th = skimage.filters.threshold_otsu(image)
    segmented = np.zeros_like(image)
    segmented[image > th] = 255
    return segmented

@click.command()
@click.argument("src", type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument("dst", type=click.Path(writable=True, dir_okay=False))
def main(src, dst):
    image = open_image(src)
    segmented = segment_image(image)
    save_image(dst, segmented)
    

if __name__ == '__main__':
    main()