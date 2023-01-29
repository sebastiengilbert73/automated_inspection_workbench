import cv2
import numpy as np

def main():
    image = cv2.imread("/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images/hens/cayenne2.jpg")
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    image = image.astype(np.float32)
    template = np.random.randn(5, 5, 3).astype(np.float32)#, 3)
    match_img = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
    print(f"match_img.shape = {match_img.shape}")

if __name__ == '__main__':
    main()