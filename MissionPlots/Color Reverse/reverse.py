import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("img313_s2_change map.jpg")
print(type(img))
print(img.shape)

# Create a copy of the image array
modified_img = img.copy()

for band in range(3):
    band_array = modified_img[:, :, band]
    band_array = 255 - band_array
    modified_img[:, :, band] = band_array

plt.imshow(modified_img)

plt.imsave("img313_s2_change map_reverse.jpg", modified_img)

plt.show()  # Don't forget to show the plot
