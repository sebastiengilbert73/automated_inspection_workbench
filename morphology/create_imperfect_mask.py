import cv2
import numpy as np
import os

output_directory = './output'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

original_img = cv2.imread('/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images/hardware/anchor_dowels1.jpg')
blue_img, green_img, red_img = cv2.split(original_img)  # Splits the 3 channels: b, g, r

# Strong green signal
green_threshold = 200
retval, green_mask = cv2.threshold(green_img, green_threshold, 255, cv2.THRESH_BINARY)
# Low red signal
red_inverse_threshold = 200
retval, red_inverse_mask = cv2.threshold(red_img, red_inverse_threshold, 255, cv2.THRESH_BINARY_INV)

# Strong green signal AND low red signal
green_dowels_mask = np.minimum(green_mask, red_inverse_mask)

# Mask the original image with the green dowels mask
green_dowels_img = np.minimum(original_img, cv2.cvtColor(green_dowels_mask, cv2.COLOR_GRAY2BGR))

# Save the images
cv2.imwrite('./output/0_original.png', original_img)
#cv2.putText(green_mask, 'green_mask', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
cv2.imwrite('./output/1_green_mask.png', green_mask)
#cv2.putText(red_inverse_mask, 'red_inverse_mask', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
cv2.imwrite('./output/2_red_inverse_mask.png', red_inverse_mask)
#cv2.putText(green_dowels_mask, 'green_dowels_mask', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
cv2.imwrite('./output/3_green_dowels_mask.png', green_dowels_mask)
#cv2.putText(green_dowels_img, 'green dowels', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
cv2.imwrite('./output/4_green_dowels.png', green_dowels_img)