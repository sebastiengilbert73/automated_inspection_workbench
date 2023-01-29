import cv2
import random
import copy

def main():
    image_filepath = "/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images_wip/fruits/bananas_black1b.jpg"

    original_img = cv2.imread(image_filepath)
    noise_prob = 0.05
    corrupted_img = copy.deepcopy(original_img)

    for y in range(original_img.shape[0]):
        for x in range(original_img.shape[1]):
            if random.random() < noise_prob:
                new_color = (int(-128 + 255 * random.random()),
                             int(-128 + 255 * random.random()),
                             int(-128 + 255 * random.random()))
                corrupted_img[y, x] = new_color

    cv2.imshow("corrupted", corrupted_img)
    cv2.waitKey(0)



if __name__ == '__main__':
    main()