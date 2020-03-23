import imageio
from skimage import filters, morphology, measure

im = imageio.imread('/Users/yoavram/Downloads/image2.jpg')
im = im[:, :, 0]
th = filters.threshold_otsu(im)
segmented = im < th
closed = morphology.binary_closing(segmented, morphology.disk(6))
labels = measure.label(closed)
props = measure.regionprops(labels)
print(max([p.major_axis_length for p in props]))