import cv2
import numpy as np
import os
import copy

def main():
    print(f"warp_perspective.main()")
    output_directory = "./output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    image_filenames = ['draw_book_perpendicular.jpg', 'draw_book_angle.jpg']
    images_directory = "./images/paper"
    # images_directory = "/usercode/images/paper"
    images = [cv2.imread(os.path.join(images_directory, filename)) for filename in image_filenames]
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    blur_size = (5, 5)
    # Blur the grayscale images
    for img_ndx in range(len(grayscale_images)):
        grayscale_images[img_ndx] = cv2.blur(grayscale_images[img_ndx], blur_size)
    # Detect four corners
    template_sizeHW = (17, 17)
    corners_NW_list = [locate_corner(grayscale_img, 'NW', template_sizeHW, blur_size) for grayscale_img in grayscale_images]
    print(f"corners_NW_list = {corners_NW_list}")
    corners_NE_list = [locate_corner(grayscale_img, 'NE', template_sizeHW, blur_size) for grayscale_img in grayscale_images]
    print(f"corners_NE_list = {corners_NE_list}")
    corners_SE_list = [locate_corner(grayscale_img, 'SE', template_sizeHW, blur_size) for grayscale_img in grayscale_images]
    print(f"corners_SE_list = {corners_SE_list}")
    corners_SW_list = [locate_corner(grayscale_img, 'SW', template_sizeHW, blur_size) for grayscale_img in
                       grayscale_images]
    print(f"corners_SW_list = {corners_SW_list}")
    # Annotate the found corners
    for img_ndx in range(len(images)):
        annotated_img = copy.deepcopy(images[img_ndx])
        cv2.circle(annotated_img, corners_NW_list[img_ndx], 5, (255, 0, 0), 2)
        cv2.circle(annotated_img, corners_NE_list[img_ndx], 5, (0, 255, 0), 2)
        cv2.circle(annotated_img, corners_SE_list[img_ndx], 5, (0, 0, 255), 2)
        cv2.circle(annotated_img, corners_SW_list[img_ndx], 5, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(output_directory, f"{img_ndx}_corners.png"), annotated_img)

    # Perspective transformation
    feature_points0 = np.array([[corners_NW_list[0][0], corners_NW_list[0][1]],
                                [corners_NE_list[0][0], corners_NE_list[0][1]],
                                [corners_SE_list[0][0], corners_SE_list[0][1]],
                                [corners_SW_list[0][0], corners_SW_list[0][1]]], dtype=np.float32)
    feature_points1 = np.array([[corners_NW_list[1][0], corners_NW_list[1][1]],
                                [corners_NE_list[1][0], corners_NE_list[1][1]],
                                [corners_SE_list[1][0], corners_SE_list[1][1]],
                                [corners_SW_list[1][0], corners_SW_list[1][1]]], dtype=np.float32)

    warped_corners = np.array([[100, 100], [1100, 100], [1100, 1100], [100, 1100]], dtype=np.float32)
    perspective_mtx0 = cv2.getPerspectiveTransform(feature_points0, warped_corners)
    perspective_mtx1 = cv2.getPerspectiveTransform(feature_points1, warped_corners)
    print(f"perspective_mtx0 =\n{perspective_mtx0}")
    print(f"perspective_mtx1 =\n{perspective_mtx1}")

    # Create a warped image
    warped_image_size = (1200, 1200)
    warped_perspective_img0 = cv2.warpPerspective(images[0], perspective_mtx0, dsize=warped_image_size)
    warped_perspective_img1 = cv2.warpPerspective(images[1], perspective_mtx1, dsize=warped_image_size)
    cv2.imwrite(os.path.join(output_directory, "0_warped_perspective.png"), warped_perspective_img0)
    cv2.imwrite(os.path.join(output_directory, "1_warped_perspective.png"), warped_perspective_img1)

def locate_corner(grayscale_img, corner, template_sizeHW, blur_size):
    template = np.zeros(template_sizeHW, dtype=np.uint8)
    template_center = (template_sizeHW[1]//2, template_sizeHW[0]//2)
    if corner == 'NW':
        cv2.rectangle(template, template_center, (template_sizeHW[1] - 1, template_sizeHW[0] - 1),
                  255, thickness=-1)
    elif corner == 'NE':
        cv2.rectangle(template, template_center, (0, template_sizeHW[0] - 1),
                      255, thickness=-1)
    elif corner == 'SE':
        cv2.rectangle(template, template_center, (0, 0),
                  255, thickness=-1)
    elif corner == 'SW':
        cv2.rectangle(template, template_center, (template_sizeHW[1] - 1, 0),
                  255, thickness=-1)
    else:
        raise NotImplementedError(f'locate_corner(): Not implemented corner {corner}')
    template = cv2.blur(template, blur_size)

    # Match template
    match_img = cv2.matchTemplate(grayscale_img, template, cv2.TM_CCOEFF_NORMED)

    # Locate the highest match
    _, max_val, _, max_loc = cv2.minMaxLoc(match_img)
    max_loc = (max_loc[0] + template_sizeHW[1] // 2, max_loc[1] + template_sizeHW[0] // 2)
    return max_loc
if __name__ == '__main__':
    main()