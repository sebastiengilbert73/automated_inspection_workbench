import copy
import os
import cv2
import numpy as np
import copy

output_directory = './output'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

template_sizeHW = (17, 17)
template_center = (template_sizeHW[1]//2, template_sizeHW[0]//2)
blur_size = (5, 5)

def locate_corner(grayscale_img, corner):
    template = np.zeros(template_sizeHW, dtype=np.uint8)
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
    cv2.imwrite(os.path.join(output_directory, f'template_{corner}.png'), template)

    # Match template
    match_img = cv2.matchTemplate(grayscale_img, template, cv2.TM_CCOEFF_NORMED)
    cv2.imwrite(os.path.join(output_directory, f'match_{corner}.png'), (128 + 127 * match_img).astype(np.uint8))
    # Locate the highest match
    _, max_val, _, max_loc = cv2.minMaxLoc(match_img)
    max_loc = (max_loc[0] + template_sizeHW[1] // 2, max_loc[1] + template_sizeHW[0] // 2)
    print(f"max_val = {max_val}; max_loc = {max_loc}")
    return max_loc

original_img = cv2.imread('./images/paper/book_draw1.jpg')
#original_img = cv2.imread('/usercode/images/paper/book_ideas4.jpg')
annotated_img = copy.deepcopy(original_img)

grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(output_directory, 'grayscale.png'), grayscale_img)


blurred_grayscale_img = cv2.blur(grayscale_img, blur_size)

for corner in ['NW', 'NE', 'SE', 'SW']:
    max_loc = locate_corner(grayscale_img, corner)
    cv2.circle(annotated_img, max_loc, radius=7, color=(255, 0, 0), thickness=2)


cv2.imwrite(os.path.join(output_directory, 'annotated.png'), annotated_img)