import cv2
import os
import numpy as np
import copy

def main():
    print("match_template.main()")

    output_directory = './output'
    ring_outer_diameter_in_pixels = 26
    ring_inner_diameter_in_pixels = 18
    match_threshold = 0.43
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    original_img = cv2.imread('/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images_wip/electronics/beaglebone.jpg')
    print(f"original_img.shape = {original_img.shape}")
    #grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Create a template
    template_shapeHWC = [round(1.5 * ring_outer_diameter_in_pixels),
                         round(1.5 * ring_outer_diameter_in_pixels), 3]
    if template_shapeHWC[0] % 2 == 0:
        template_shapeHWC[0] += 1
    if template_shapeHWC[1] % 2 == 0:
        template_shapeHWC[1] += 1
    print(f"template_shapeHWC = {template_shapeHWC}")
    template_img = np.zeros(template_shapeHWC, dtype=np.uint8)
    # Draw the ring: A yellow disk, then a smaller black disk
    cv2.circle(template_img, center=(template_shapeHWC[1]//2, template_shapeHWC[0]//2), radius=ring_outer_diameter_in_pixels//2,
               color=(128, 210, 255), thickness=-1)
    cv2.circle(template_img, center=(template_shapeHWC[1] // 2, template_shapeHWC[0] // 2), radius=ring_inner_diameter_in_pixels//2,
               color=0, thickness=-1)

    # Match template
    match_img = cv2.matchTemplate(original_img, template_img, method=cv2.TM_CCOEFF_NORMED)
    print(f"match_img.dtype = {match_img.dtype}")
    print(f"match_img.shape = {match_img.shape}")

    # Print the match range of values
    minimum_match_value = np.min(match_img)
    maximum_match_value = np.max(match_img)
    print(f"minimum_match_value = {minimum_match_value}; maximum_match_value = {maximum_match_value}")

    # Convert the match image to an 8-bit image
    match_8b_img = (127.0 + 128.0 * match_img).astype(np.uint8)

    # Zero-pad the match image, such that it's size is identical to that of the original image
    zero_padded_match_img = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=float)
    zero_padded_match_img[template_shapeHWC[0]//2: template_shapeHWC[0]//2 + match_img.shape[0],
        template_shapeHWC[1]//2: template_shapeHWC[1]//2 + match_img.shape[1]] = match_img

    # Threshold the match image
    retval, thresholded_match_img = cv2.threshold(zero_padded_match_img, match_threshold, 255, cv2.THRESH_BINARY)

    # Annotate the original image
    annotated_img = copy.deepcopy(original_img)
    for y in range(thresholded_match_img.shape[0]):
        for x in range(thresholded_match_img.shape[1]):
            if thresholded_match_img[y, x] > 0:
                annotated_img[y, x] = (0, 0, 255)

    cv2.imwrite(os.path.join(output_directory, 'original.png'), original_img)
    #cv2.imwrite(os.path.join(output_directory, 'grayscale.png'), grayscale_img)
    cv2.imwrite(os.path.join(output_directory, 'template.png'), template_img)
    cv2.imwrite(os.path.join(output_directory, 'match.png'), match_8b_img)
    cv2.imwrite(os.path.join(output_directory, 'thresholded_match.png'), thresholded_match_img)
    cv2.imwrite(os.path.join(output_directory, 'annotated.png'), annotated_img)

if __name__ == '__main__':
    main()