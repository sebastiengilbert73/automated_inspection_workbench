import cv2
import os
import numpy as np

mask = cv2.imread('./output/3_green_dowels_mask.png', cv2.IMREAD_GRAYSCALE)
original_img = cv2.imread('/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images/hardware/anchor_dowels1.jpg')

dilation_erosion_kernel = np.ones((7, 7), dtype=np.uint8)
dilated_mask = cv2.dilate(mask, dilation_erosion_kernel)
cv2.imwrite('./output/dilated_mask.png', dilated_mask)

dilated_eroded_mask = cv2.erode(dilated_mask, dilation_erosion_kernel)
cv2.imwrite('./output/dilated_eroded_mask.png', dilated_eroded_mask)

erosion_dilation_kernel = np.ones((3, 3), dtype=np.uint8)
eroded_mask = cv2.erode(dilated_eroded_mask, erosion_dilation_kernel)
cv2.imwrite('./output/eroded_mask.png', eroded_mask)

eroded_dilated_mask = cv2.dilate(eroded_mask, erosion_dilation_kernel)
cv2.imwrite('./output/eroded_dilated.png', eroded_dilated_mask)

green_dowels_img = np.minimum(original_img, cv2.cvtColor(eroded_dilated_mask, cv2.COLOR_GRAY2BGR))
cv2.imwrite('./output/green_dowels.png', green_dowels_img)
