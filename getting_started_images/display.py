import cv2

def main():
    img_filepath = "/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images/hens/paprika2.jpg"
    image = cv2.imread(img_filepath)
    cv2.imshow("Paprika", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()