import cv2
import numpy as np
import os

def main():
    print("simulate_line.main()")
    output_directory = "./output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    img_shapeHWC = (480, 640, 3)
    line_thickness = 3
    blurring_sizeWH = (1, 7)
    noise_amplitude = 10
    simulated_img = np.zeros(img_shapeHWC, dtype=np.uint8)
    zero_level = (450, 0.05)
    p0 = (100, 0)
    p1 = (150, 0)
    p2 = (175, 300)
    p3 = (200, 300)
    p4 = (300, 150)
    p5 = (350, 170)
    p6 = (400, 300)
    p7 = (460, 300)
    p8 = (500, 0)
    p9 = (550, 0)
    points = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    for pt_ndx in range(len(points) - 1):
        q1 = points[pt_ndx]
        q2 = points[pt_ndx + 1]
        q1_ = (q1[0], round(zero_level[0] - zero_level[1] * q1[0] - q1[1]))
        q2_ = (q2[0], round(zero_level[0] - zero_level[1] * q2[0] - q2[1]))
        cv2.line(simulated_img, q1_, q2_, (0, 0, 255), thickness=line_thickness)
    random_modulation = np.random.random((img_shapeHWC[0], img_shapeHWC[1]))
    simulated_img[:, :, 2] = (simulated_img[:, :, 2] * random_modulation).astype(np.uint8)
    simulated_img = cv2.blur(simulated_img, blurring_sizeWH)
    noise_img = noise_amplitude * np.random.random(img_shapeHWC)
    simulated_img = np.clip(simulated_img.astype(np.float32) + noise_img, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_directory, "simulated.png"), simulated_img)


if __name__ == '__main__':
    main()