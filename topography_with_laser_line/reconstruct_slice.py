import cv2
import numpy as np
import os
import simulate_slice_image
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("reconstruct_slice.main()")
    output_directory = "./output_reconstruct_slice"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Obtain a slice image
    topography_arr = np.load("./data/topography.npy")
    laser_reachable_mask = cv2.imread("./data/laser_reachable_mask.jpg", cv2.IMREAD_GRAYSCALE)
    calibration_df = pd.read_csv("./data/calib.csv")
    slice_img = simulate_slice_image.main(
        topography_arr,
        laser_reachable_mask,
        calibration_df,
        yInTopography=300,
        img_sizeHW=(480, 640),
        lineThickness=3,
        blurring_sizeWH=(1, 7),
        noiseAmplitude=10,
        saveImages=False
    )

    blurred_red_img = cv2.blur(slice_img[:, :, 2], (5, 5))
    # For each x, locate the strongest y, if above a threshold
    heights = []
    minimum_red_value = 20
    for x in range(blurred_red_img.shape[1]):
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred_red_img[:, x])
        if max_val >= minimum_red_value:
            y = max_loc[1]  # (0, y_max)
            a = calibration_df.loc[calibration_df['x'] == x].iloc[0]['a']
            b = calibration_df.loc[calibration_df['x'] == x].iloc[0]['b']
            heights.append(a * y + b)
        else:
            heights.append(-1)  # Special flag for 'unknown'
    xs = np.arange(0, len(heights))
    fig, ax = plt.subplots()
    ax.scatter(xs, heights, c="blue", marker='.')
    ax.set_xlabel('x')
    ax.set_ylabel('height (mm)')
    plt.savefig(os.path.join(output_directory, "slice_in_mm.png"))


    cv2.imwrite(os.path.join(output_directory, "slice.png"), slice_img)
    cv2.imwrite(os.path.join(output_directory, "blurred_red.png"), blurred_red_img)

if __name__ == '__main__':
    main()