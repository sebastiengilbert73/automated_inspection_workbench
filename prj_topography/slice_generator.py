import simulate_slice_image
import numpy as np
import cv2
import pandas as pd

def get_image(y, x_to_a_b):
    topography_arr = np.load("/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images/topography/topography.npy")
    # topography_arr = np.load("/usercode/images/topography/topography.npy")
    laser_reachable_mask = cv2.imread("/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images/topography/laser_reachable_mask.jpg", cv2.IMREAD_GRAYSCALE)
    # laser_reachable_mask = cv2.imread("/usercode/images/topography/laser_reachable_mask.jpg")
    x_list = list(x_to_a_b.keys())
    ab_list = list(x_to_a_b.values())
    a_list = [ab_list[i][0] for i in range(len(ab_list))]
    b_list = [ab_list[i][1] for i in range(len(ab_list))]
    calibration_df = pd.DataFrame({'x': x_list, 'a': a_list, 'b': b_list})
    slice_img = simulate_slice_image.main(
        topography_arr,
        laser_reachable_mask,
        calibration_df,
        y,
        (480, 640),
        3,
        (3, 7),
        10,
        False
    )
    return slice_img