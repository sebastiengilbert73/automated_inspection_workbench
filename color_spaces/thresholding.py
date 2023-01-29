import cv2

def main():
    image = cv2.imread(
        "/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images/hens/cayenne2.jpg")
    retval, thresholded_img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite('./thresholded.png', thresholded_img)
if __name__ == '__main__':
    main()