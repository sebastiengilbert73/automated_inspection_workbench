import cv2
import numpy as np
import os
import argparse
import pandas as pd
import ast

def main(
    topography_arr,
    laser_reachable_mask,
    calibration_df,
    yInTopography,
    img_sizeHW,
    lineThickness,
    blurring_sizeWH,
    noiseAmplitude,
    saveImages
):
    print("simulate_slice_image.main()")

    simulated_slice_img = np.zeros((img_sizeHW[0], img_sizeHW[1], 3), dtype=np.uint8)

    for x in range(img_sizeHW[1]):
        calib_a = calibration_df.loc[calibration_df['x'] == x]['a'].values[0]
        calib_b = calibration_df.loc[calibration_df['x'] == x]['b'].values[0]
        if laser_reachable_mask[yInTopography, x] > 0 and calib_a != 0:
            height = topography_arr[yInTopography, x]
            y = round((height - calib_b)/calib_a)
            simulated_slice_img[y, x, :] = (0, 0, 240)
    simulated_slice_img = cv2.dilate(simulated_slice_img, np.ones((lineThickness, 1), dtype=np.uint8))

    random_modulation = np.random.random((img_sizeHW[0], img_sizeHW[1]))
    simulated_slice_img[:, :, 2] = (simulated_slice_img[:, :, 2] * random_modulation).astype(np.uint8)
    simulated_slice_img = cv2.blur(simulated_slice_img, blurring_sizeWH)
    noise_img = noiseAmplitude * np.random.random((img_sizeHW[0], img_sizeHW[1], 3))
    simulated_slice_img = np.clip(simulated_slice_img.astype(np.float32) + noise_img, 0, 255).astype(np.uint8)

    if saveImages:
        cv2.imwrite('./simulated_slice.png', simulated_slice_img)

    return simulated_slice_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("topographyFilepath", help="The filepath to the topography numpy array"),
    parser.add_argument("reachableMaskFilepath", help="The filepath to the mask of laser reachable area")
    parser.add_argument("calibrationFilepath", help="The filepath to the csv calibration file")
    parser.add_argument("yInTopography", help="The y coordinate in the topography array", type=int)
    parser.add_argument("--cameraResolution", help="The camera resolution (H, W). Default: '(480, 640)'", default='(480, 640)')
    parser.add_argument("--lineThickness", help="The laser line thickness. Default: 3", type=int, default=3)
    parser.add_argument("--blurringSize", help="The blurring size (W, H). Default: '(1, 7)'", default='(1, 7)')
    parser.add_argument("--noiseAmplitude", help="The random noise amplitude. Default: 10", type=float, default=10)
    parser.add_argument("--saveImages", help="Save the intermediary images", action='store_true')
    args = parser.parse_args()
    topography_arr = np.load(args.topographyFilepath)
    laser_reachable_mask = cv2.imread(args.reachableMaskFilepath, cv2.IMREAD_GRAYSCALE)
    calibration_df = pd.read_csv(args.calibrationFilepath)
    img_sizeHW = ast.literal_eval(args.cameraResolution)
    blurring_sizeWH = ast.literal_eval(args.blurringSize)
    main(
        topography_arr,
        laser_reachable_mask,
        calibration_df,
        args.yInTopography,
        img_sizeHW,
        args.lineThickness,
        blurring_sizeWH,
        args.noiseAmplitude,
        args.saveImages
    )