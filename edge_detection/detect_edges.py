import cv2
import numpy as np
import os
import copy
import math

def main():
    output_directory = './output'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    original_img = cv2.imread('./images/electronics/rpi_back.jpg')
    cv2.imwrite(os.path.join(output_directory, 'original.png'), original_img)
    blue_img, green_img, red_img = cv2.split(original_img)
    cv2.imwrite(os.path.join(output_directory, '0_blue.png'), blue_img)
    cv2.imwrite(os.path.join(output_directory, '1_green.png'), green_img)
    cv2.imwrite(os.path.join(output_directory, '2_red.png'), red_img)

    # We choose to use the red image

    # Create a mask of low red signal
    low_red_inverse_threshold = 36
    retval, low_red_mask = cv2.threshold(red_img, low_red_inverse_threshold, 255,
                                         cv2.THRESH_BINARY_INV)
    # Erode and dilate the mask
    erosion_dilation_kernel = np.ones((3, 3), dtype=np.uint8)
    low_red_mask = cv2.erode(low_red_mask, erosion_dilation_kernel)
    low_red_mask = cv2.dilate(low_red_mask, erosion_dilation_kernel)
    # Dilate a few more times to include a bit more area in the periphery
    low_red_mask = cv2.dilate(low_red_mask, erosion_dilation_kernel, iterations=5)
    cv2.imwrite(os.path.join(output_directory, 'low_red_mask.png'), low_red_mask)

    # Blur the red channel image
    blurred_red_img = cv2.blur(red_img, (5, 5))

    # Laplacian
    laplacian_img = cv2.Laplacian(blurred_red_img, ddepth=cv2.CV_32F)
    print(f"np.min(laplacian_img) = {np.min(laplacian_img)}; np.max(laplacian_img) = {np.max(laplacian_img)}")
    laplacian_threshold = 10
    retval, thresholded_laplacian_img = cv2.threshold(laplacian_img, laplacian_threshold, 255, cv2.THRESH_BINARY)
    thresholded_laplacian_img = np.minimum(thresholded_laplacian_img, low_red_mask)
    cv2.imwrite(os.path.join(output_directory, 'thresholded_laplacian.png'), thresholded_laplacian_img)
    # Mask the laplacian image with the low red mask
    laplacian_img = np.minimum(laplacian_img, low_red_mask.astype(float))
    cv2.imwrite(os.path.join(output_directory, '3_laplacian.png'), laplacian_img)

    # Sobel
    sobel_x_img = cv2.Sobel(blurred_red_img, ddepth=cv2.CV_32F, dx=1, dy=0)
    sobel_y_img = cv2.Sobel(blurred_red_img, ddepth=cv2.CV_32F, dx=0, dy=1)
    cv2.imwrite(os.path.join(output_directory, '4_sobel_x.png'), sobel_x_img)
    cv2.imwrite(os.path.join(output_directory, '5_sobel_y.png'), sobel_y_img)
    print(f"np.min(sobel_x_img) = {np.min(sobel_x_img)}; np.max(sobel_x_img) = {np.max(sobel_x_img)}")
    print(f"np.min(sobel_y_img) = {np.min(sobel_y_img)}; np.max(sobel_y_img) = {np.max(sobel_y_img)}")
    # Threshold the Sobel images
    sobel_threhold = 130
    retval, high_positive_sobel_x_img = cv2.threshold(sobel_x_img, sobel_threhold, 255, cv2.THRESH_BINARY)
    high_positive_sobel_x_img = np.minimum(high_positive_sobel_x_img, low_red_mask).astype(np.uint8)
    cv2.imwrite(os.path.join(output_directory, 'high_positive_sobel_x.png'), high_positive_sobel_x_img)
    retval, high_negative_sobel_x_img = cv2.threshold(sobel_x_img, -sobel_threhold, 255, cv2.THRESH_BINARY_INV)
    high_negative_sobel_x_img = np.minimum(high_negative_sobel_x_img, low_red_mask).astype(np.uint8)
    cv2.imwrite(os.path.join(output_directory, 'high_negative_sobel_x.png'), high_negative_sobel_x_img)

    retval, high_positive_sobel_y_img = cv2.threshold(sobel_y_img, sobel_threhold, 255, cv2.THRESH_BINARY)
    high_positive_sobel_y_img = np.minimum(high_positive_sobel_y_img, low_red_mask).astype(np.uint8)
    cv2.imwrite(os.path.join(output_directory, 'high_positive_sobel_y.png'), high_positive_sobel_y_img)
    retval, high_negative_sobel_y_img = cv2.threshold(sobel_y_img, -sobel_threhold, 255, cv2.THRESH_BINARY_INV)
    high_negative_sobel_y_img = np.minimum(high_negative_sobel_y_img, low_red_mask).astype(np.uint8)
    cv2.imwrite(os.path.join(output_directory, 'high_negative_sobel_y.png'), high_negative_sobel_y_img)

    # Canny
    canny_threshold1 = 100
    canny_threshold2 = 50
    canny_img = cv2.Canny(blurred_red_img, canny_threshold1, canny_threshold2)
    canny_img = np.minimum(canny_img, low_red_mask)
    cv2.imwrite(os.path.join(output_directory, 'canny.png'), canny_img)

    # Thinning the thresholded Sobel images
    thinned_sobel_west_img = cv2.ximgproc.thinning(high_negative_sobel_x_img)
    cv2.imwrite(os.path.join(output_directory, 'thinned_sobel_west.png'), thinned_sobel_west_img)
    thinned_sobel_east_img = cv2.ximgproc.thinning(high_positive_sobel_x_img)
    cv2.imwrite(os.path.join(output_directory, 'thinned_sobel_east.png'), thinned_sobel_east_img)
    thinned_sobel_north_img = cv2.ximgproc.thinning(high_negative_sobel_y_img)
    cv2.imwrite(os.path.join(output_directory, 'thinned_sobel_north.png'), thinned_sobel_north_img)
    thinned_sobel_south_img = cv2.ximgproc.thinning(high_positive_sobel_y_img)
    cv2.imwrite(os.path.join(output_directory, 'thinned_sobel_south.png'), thinned_sobel_south_img)


    # Hough line detection
    rho_resolution = 1.0
    theta_resolution = 0.001
    hough_accumulator_threshold = 150
    lines_west = cv2.HoughLines(thinned_sobel_west_img, rho=rho_resolution, theta=theta_resolution,
                                  threshold=hough_accumulator_threshold)
    print(f"lines_west = {lines_west}")
    lines_east = cv2.HoughLines(thinned_sobel_east_img, rho=rho_resolution, theta=theta_resolution,
                                  threshold=hough_accumulator_threshold)
    print(f"lines_east = {lines_east}")
    lines_north = cv2.HoughLines(thinned_sobel_north_img, rho=rho_resolution, theta=theta_resolution,
                                 threshold=hough_accumulator_threshold)
    print(f"lines_north = {lines_north}")
    lines_south = cv2.HoughLines(thinned_sobel_south_img, rho=rho_resolution, theta=theta_resolution,
                                   threshold=hough_accumulator_threshold)
    print(f"lines_south = {lines_south}")

    annotated_img = copy.deepcopy(original_img)
    DrawLine(annotated_img, lines_west[0][0], (255, 0, 0))
    DrawLine(annotated_img, lines_north[0][0], (0, 255, 0))
    DrawLine(annotated_img, lines_east[0][0], (0, 0, 255))
    DrawLine(annotated_img, lines_south[0][0], (0, 255, 255))

    cv2.imwrite(os.path.join(output_directory, 'annotated.png'), annotated_img)

def DrawLine(image, rho_theta, color):
    rho = rho_theta[0]
    theta = rho_theta[1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = round(x0 - 1000 * b)
    y1 = round(y0 + 1000 * a)
    x2 = round(x0 + 1000 * b)
    y2 = round(y0 - 1000 * a)
    cv2.line(image, (x1, y1), (x2, y2), color=color, thickness=1)
if __name__ == '__main__':
    main()