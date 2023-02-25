import cv2
import os
import numpy as np
import copy

def main():
    output_directory = './output'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    original1_img = cv2.imread("./images/paper/book_draw1.jpg")
    original2_img = cv2.imread("./images/paper/book_draw2.jpg")
    #original_img = cv2.imread("/usercode/images/electronics/nano.jpg")
    grayscale1_img = cv2.cvtColor(original1_img, cv2.COLOR_BGR2GRAY)
    grayscale2_img = cv2.cvtColor(original2_img, cv2.COLOR_BGR2GRAY)

    source_window_name = 'Source image'
    corners_window_name = 'Corners detected'

    block_size = 2
    aperture_size = 3
    k = 0.04
    corners_threshold = 0.001

    # Detecting corners
    corners1_img = cv2.cornerHarris(grayscale1_img, block_size, aperture_size, k)
    print(f"np.min(corners1_img) = {np.min(corners1_img)}; np.max(corners1_img) = {np.max(corners1_img)}")
    retval, thresholded_corners1_img = cv2.threshold(corners1_img, corners_threshold, 255, cv2.THRESH_BINARY)
    thresholded_corners1_img_filepath = os.path.join(output_directory, "thresholded_corners1.png")
    cv2.imwrite(thresholded_corners1_img_filepath, thresholded_corners1_img)

    corners2_img = cv2.cornerHarris(grayscale2_img, block_size, aperture_size, k)
    print(f"np.min(corners2_img) = {np.min(corners2_img)}; np.max(corners2_img) = {np.max(corners2_img)}")
    retval, thresholded_corners2_img = cv2.threshold(corners2_img, corners_threshold, 255, cv2.THRESH_BINARY)
    thresholded_corners2_img_filepath = os.path.join(output_directory, "thresholded_corners2.png")
    cv2.imwrite(thresholded_corners2_img_filepath, thresholded_corners2_img)

    # Good features to track
    max_corners = 30
    quality_level = 0.01
    min_distance = 10
    gft1_corners = cv2.goodFeaturesToTrack(grayscale1_img, max_corners, quality_level, min_distance).astype(int)
    print(f"gft1_corners = {gft1_corners}")
    print(f"gft1_corners.shape = {gft1_corners.shape}")
    annotated_gft1_img = copy.deepcopy(original1_img)
    for corner_ndx in range(gft1_corners.shape[0]):
        cv2.circle(annotated_gft1_img, gft1_corners[corner_ndx][0], 5, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_directory, "annotated_gft1.png"), annotated_gft1_img)

    gft2_corners = cv2.goodFeaturesToTrack(grayscale2_img, max_corners, quality_level, min_distance).astype(int)
    print(f"gft2_corners = {gft2_corners}")
    print(f"gft2_corners.shape = {gft2_corners.shape}")
    annotated_gft2_img = copy.deepcopy(original2_img)
    for corner_ndx in range(gft2_corners.shape[0]):
        cv2.circle(annotated_gft2_img, gft2_corners[corner_ndx][0], 5, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_directory, "annotated_gft2.png"), annotated_gft2_img)


if __name__ == '__main__':
    main()