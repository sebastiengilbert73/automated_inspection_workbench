import cv2
import random
import copy
import numpy as np

def main():
    image_filepath = "/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images_wip/cans/diced_tomatoes.jpg"

    original_img = cv2.imread(image_filepath)
    noise_std_dev = 30.0
    corrupted_img = copy.deepcopy(original_img)

    noise = np.random.normal(0, noise_std_dev, original_img.shape)
    for y in range(original_img.shape[0]):
        for x in range(original_img.shape[1]):
            original_color = original_img[y, x, :]
            new_color = (np.clip(original_color[0] + round(noise[y, x, 0]), 0, 255),
                         np.clip(original_color[1] + round(noise[y, x, 1]), 0, 255),
                         np.clip(original_color[2] + round(noise[y, x, 2]), 0, 255))
            corrupted_img[y, x] = new_color

    cv2.imshow("corrupted", corrupted_img)
    cv2.waitKey(0)



if __name__ == '__main__':
    main()