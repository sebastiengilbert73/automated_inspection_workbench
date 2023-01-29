import cv2

def main():
    image = cv2.imread('/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images_wip/paper/text_gradient.jpg',
                       cv2.IMREAD_GRAYSCALE)
    thresholded_img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 20)
    cv2.imshow("mask", thresholded_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()