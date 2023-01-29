import cv2
import random
import copy
import numpy as np

def main():
    print("smooth_mask.main()")

    original_img = cv2.imread("/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images_wip/cans/diced_tomatoes_noise.jpg")

    blurred_img = cv2.blur(original_img, (7, 7))
    blue_img, green_img, red_img = cv2.split(blurred_img)

    # Threshold the blue channel
    retval, thresholded_blue_img = cv2.threshold(blue_img, 80, 255, cv2.THRESH_BINARY)

    # Threshold the red channel
    retval, thresholded_red_img = cv2.threshold(red_img, 80, 255, cv2.THRESH_BINARY)

    # Union of the blue and the red masks
    mask = np.maximum(thresholded_blue_img, thresholded_red_img)

    # Mask the original image
    masked_cans_img = np.minimum(original_img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    cv2.imwrite('./output/blue.png', blue_img)
    cv2.imwrite('./output/green.png', green_img)
    cv2.imwrite('./output/red.png', red_img)
    cv2.imwrite('./output/blurred_7.png', blurred_img)
    cv2.imwrite('./output/thresholded_blue.png', thresholded_blue_img)
    cv2.imwrite('./output/thresholded_red.png', thresholded_red_img)
    cv2.imwrite('./output/mask.png', mask)
    cv2.imwrite('./output/masked_cans.png', masked_cans_img)


if __name__ == '__main__':
    main()