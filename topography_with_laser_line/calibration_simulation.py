import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    print("calibration_simulation.main()")
    output_directory = "./output_calibration_simulation"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    slice_img_sizeHW = (480, 640)

    slice_20_img = simulate_image(slice_img_sizeHW, 300, -0.03, 3, (3, 7), 20)
    slice_40_img = simulate_image(slice_img_sizeHW, 200, -0.012, 3, (3, 7), 20)
    slice_60_img = simulate_image(slice_img_sizeHW, 100, 0.006, 3, (3, 7), 20)
    cv2.imwrite(os.path.join(output_directory, "slice_20.png"), slice_20_img)
    cv2.imwrite(os.path.join(output_directory, "slice_40.png"), slice_40_img)
    cv2.imwrite(os.path.join(output_directory, "slice_60.png"), slice_60_img)

    # Blur the red channel of each image
    blur_size = (5, 5)
    blurred_slice_20_img = cv2.blur(slice_20_img[:, :, 2], blur_size)
    blurred_slice_40_img = cv2.blur(slice_40_img[:, :, 2], blur_size)
    blurred_slice_60_img = cv2.blur(slice_60_img[:, :, 2], blur_size)

    minimum_red_value = 10
    x_to_ymax_20 = {}
    x_to_ymax_40 = {}
    x_to_ymax_60 = {}
    for x in range(slice_img_sizeHW[1]):
        # Extract a single column of each blurred image
        column_20_img = blurred_slice_20_img[:, x]
        column_40_img = blurred_slice_40_img[:, x]
        column_60_img = blurred_slice_60_img[:, x]
        _, max_val_20, _, max_loc_20 = cv2.minMaxLoc(column_20_img)
        _, max_val_40, _, max_loc_40 = cv2.minMaxLoc(column_40_img)
        _, max_val_60, _, max_loc_60 = cv2.minMaxLoc(column_60_img)
        if max_val_20 > minimum_red_value:
            x_to_ymax_20[x] = max_loc_20[1]  # We keep the y value
        else:
            x_to_ymax_20[x] = -1  # Flag to indicate that the signal is too weak
        if max_val_40 > minimum_red_value:
            x_to_ymax_40[x] = max_loc_40[1]  # We keep the y value
        else:
            x_to_ymax_40[x] = -1  # Flag to indicate that the signal is too weak
        if max_val_60 > minimum_red_value:
            x_to_ymax_60[x] = max_loc_60[1]  # We keep the y value
        else:
            x_to_ymax_60[x] = -1  # Flag to indicate that the signal is too weak

    # For each x, fit a line that best satisfies the three points (y_max, Y) where Y is the height in mm
    x_to_a_b = {}
    for x in range(slice_img_sizeHW[1]):
        if x_to_ymax_20[x] >= 0 and x_to_ymax_40[x] >= 0 and x_to_ymax_60[x] >= 0:
            p0 = (x_to_ymax_20[x], 20.0)
            p1 = (x_to_ymax_40[x], 40.0)
            p2 = (x_to_ymax_60[x], 60.0)
            # Solve an overdetermined system of linear equations:
            #      A x = b
            # | x_0   1 | | a |   | y_0 |
            # | x_1   1 | | b | = | y_1 |
            # | x_2   1 |         | y_2 |
            A = np.ones((3, 2), dtype=float)
            b = np.zeros((3, 1), dtype=float)
            A[0, 0] = p0[0]
            A[1, 0] = p1[0]
            A[2, 0] = p2[0]
            b[0, 0] = p0[1]
            b[1, 0] = p1[1]
            b[2, 0] = p2[1]
            ab, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            a = ab[0, 0]
            b = ab[1, 0]
            x_to_a_b[x] = (a, b)
        else:
            x_to_a_b[x] = (None, None)

    # Plot the a, b parameters
    x_list = list(range(slice_img_sizeHW[1]))
    a_list = [x_to_a_b[x][0] for x in x_list]
    b_list = [x_to_a_b[x][1] for x in x_list]
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x_list, a_list)
    axs[0].set_xlim(0, slice_img_sizeHW[1] - 1)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('a')
    axs[0].grid(True)
    axs[1].plot(x_list, b_list)
    axs[1].set_xlim(0, slice_img_sizeHW[1] - 1)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('b')
    axs[1].grid(True)
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(output_directory, "a_b.png"))


def simulate_image(image_sizeHW, line_y, slope, line_thickness, blurring_sizeWH, noise_amplitude):
    slice_img = np.zeros((image_sizeHW[0], image_sizeHW[1], 3), dtype=np.uint8)
    p1 = (0, line_y)
    p2 = (image_sizeHW[1] - 1, round(line_y + slope * (image_sizeHW[1] - 1)))
    cv2.line(slice_img, p1, p2, (0, 0, 180), line_thickness)
    slice_img = cv2.blur(slice_img, blurring_sizeWH)
    noise = noise_amplitude * np.random.random(slice_img.shape)
    slice_img = np.clip(slice_img.astype(float) + noise, 0, 255).astype(np.uint8)
    return slice_img


if __name__ == '__main__':
    main()