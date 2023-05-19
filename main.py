
import cv2
import numpy as np
#importing required libraries
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt


A=cv2.imread("pen.png")
fd, hog_image =  hog(A, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
# cv.imshow("kir mamad",hog_image_rescaled)
# cv.waitKey(0)
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()
print(hog_image_rescaled)
print(fd)
cv2.imshow("kir",hog_image_rescaled)
cv2.waitKey(0)
cv2.imshow("kir",fd)
cv2.waitKey(0)

