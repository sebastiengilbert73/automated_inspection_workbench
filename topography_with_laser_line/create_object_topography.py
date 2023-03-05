import cv2
import numpy as np
import os
import imutils

def main():
    print("create_object_topography.py")
    output_directory = './output_create_object_topography'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    img_sizeHW = (640, 640)
    blur_size = (11, 11)

    object_img = np.zeros((img_sizeHW[0], img_sizeHW[1]), dtype=np.float32)
    mask = np.zeros((img_sizeHW[0], img_sizeHW[1]), dtype=np.uint8)

    corners = [[200, 80], [440, 80], [440, 560], [200, 560]]
    ramp_corners = [[240, 120], [400, 120], [400, 300], [240, 300]]
    pillar_center = (350, 450)
    pillar_radius = 50
    noise_std_dev = 2.0
    rotation_in_degrees = -20

    cv2.rectangle(object_img, corners[0], corners[2], 60.0, thickness=-1)
    cv2.rectangle(mask, corners[0], corners[2], 255, thickness=-1)

    # Ramp
    ramp_img = 60.0 * np.ones((ramp_corners[3][1] - ramp_corners[0][1], ramp_corners[1][0] - ramp_corners[0][0]), dtype=np.float32)
    for x in range(ramp_img.shape[1]):
        z = 60 - x/ramp_img.shape[1] * 20
        ramp_img[:, x] = z
    object_img[ramp_corners[0][1]: ramp_corners[3][1], ramp_corners[0][0]: ramp_corners[1][0]] = ramp_img

    # Pillar
    cv2.circle(object_img, pillar_center, pillar_radius, 80.0, thickness=-1)

    # Random noise
    noise_img = np.random.normal(0.0, noise_std_dev, (img_sizeHW[0]//10, img_sizeHW[1]//10))
    noise_img = cv2.resize(noise_img, img_sizeHW)
    object_img = np.clip(object_img + noise_img, 0, 100.0)

    # Rotation
    object_img = imutils.rotate(object_img, rotation_in_degrees)
    mask = imutils.rotate(mask, rotation_in_degrees)

    object_img = cv2.blur(object_img, blur_size)  # Round every edge

    # Mask
    object_img = np.minimum(object_img, mask.astype(np.float32))

    # Create an 8 bit image
    object_8bimg = object_img.astype(np.uint8)

    np.save(os.path.join(output_directory, "topography.npy"), object_img)
    cv2.imwrite(os.path.join(output_directory, "topography.png"), object_8bimg)

if __name__ == '__main__':
    main()