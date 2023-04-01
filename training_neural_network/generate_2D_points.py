import numpy as np
import os
import logging
import cv2
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main():
    logging.info("generate_2D_points.main()")
    output_directory = "./output_generate_2D_points"
    if not os.path.exists((output_directory)):
        os.makedirs(output_directory)

    number_of_points = 1000

    # Create a decision map
    decision_map = -1 * np.ones((512, 512), dtype=float)
    centers_positive = [(50, 50), (300, 200), (450, 30), (250, 50), (450, 150)]
    #centers_negative = [(40, 100), (350, 400)]#, (400, 100)]
    for center in centers_positive:
        cv2.circle(decision_map, center, 100, 1.0, thickness=-1)

    dilation_kernel = np.where(np.random.random((39, 39)) > 0.5, 1, 0).astype(np.uint8)
    decision_map = cv2.dilate(decision_map, dilation_kernel)

    color_decision_map = np.zeros((512, 512, 3), dtype=np.uint8)
    color_decision_map[:, :, 0] = np.where(decision_map >= 0, 255, 0)
    color_decision_map[:, :, 2] = np.where(decision_map < 0, 255, 0)

    color_decision_map_filepath = os.path.join(output_directory, "color_decision_map.png")
    cv2.imwrite(color_decision_map_filepath, color_decision_map)

    # draw random numbers
    class_0_points = []
    class_1_points = []
    with open(os.path.join(output_directory, "dataset.csv"), 'w') as output_file:
        output_file.write("v0,v1,class\n")
        for pt_ndx in range(number_of_points):
            l1 = np.random.randn()
            l2 = np.random.randn()
            x = np.clip(127 + round(40 * l1), 0, decision_map.shape[1] - 1)
            y = np.clip(127 + round(40 * l2), 0, decision_map.shape[0] - 1)
            class_ndx = 1
            if decision_map[y, x] < 0:
                class_ndx = 0
            # Move the point randomly to blur the boundary
            l1 += 0.2 * np.random.randn()
            l2 += 0.2 * np.random.randn()
            output_file.write(f"{l1},{l2},{class_ndx}\n")
            if class_ndx == 0:
                class_0_points.append((l1, l2))
            else:
                class_1_points.append((l1, l2))
    class_0_xs = [x for (x, y) in class_0_points]
    class_0_ys = [y for (x, y) in class_0_points]
    class_1_xs = [x for (x, y) in class_1_points]
    class_1_ys = [y for (x, y) in class_1_points]
    fig, ax = plt.subplots()
    scatter0 = ax.scatter(class_0_xs, class_0_ys, c='r')
    scatter1 = ax.scatter(class_1_xs, class_1_ys, c='b')
    ax.set_xlabel('v0')
    ax.set_ylabel('v1')
    plt.show()

if __name__ == '__main__':
    main()