import cv2
import numpy as np

block_size = 9
img_sizeHW = (block_size, block_size)
C = -10
image = 175 * np.ones(img_sizeHW, dtype=np.uint8)

for central_value in range(150, 220):
    image[img_sizeHW[0]//2, img_sizeHW[1]//2] = central_value
    thresholded_img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, block_size, C)
    number_of_active_pixels = cv2.countNonZero(thresholded_img)
    print(f"central_value = {central_value}; number_of_active_pixels = {number_of_active_pixels}")