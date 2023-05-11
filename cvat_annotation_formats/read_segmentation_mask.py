import os
import torch
import ast
import cv2
import numpy as np

def main():
    print("read_segmentation_mask.main()")

    output_directory = "./output_read_segmentation_mask"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Read the mapping between the category and the color
    labelmap_filepath = "./images/microcontrollers_segmentation_mask/labelmap.txt"

    lines = None
    with open(labelmap_filepath, 'r') as labelmap_file:
        lines = labelmap_file.readlines()
    color_to_categoryIndex = {}
    category_to_index = {'background': 0, 'capacitor': 1, 'diode': 2, 'main chip': 3, 'substrate': 4}
    for line_ndx in range(1, len(lines)):  # We skip the 1st line, as it is a comment
        # label:color_rgb:parts:actions
        label_colorRgb_parts_actions = lines[line_ndx].split(':')  # Split on ':'
        print(f"label_colorRgb_parts_actions = {label_colorRgb_parts_actions}")
        label = label_colorRgb_parts_actions[0]  # Label is synonymous to category, in this context
        color = ast.literal_eval(label_colorRgb_parts_actions[1])  # Convert a string to a tuple: '0,0,0' -> (0, 0, 0)
        color_to_categoryIndex[color] = category_to_index[label]
    print(f"color_to_categoryIndex = {color_to_categoryIndex}")

    # Read the semantic segmentation color images
    color_semantic_segmentation_directory = "./images/microcontrollers_segmentation_mask/SegmentationClass"
    color_semantic_segmentation_filepaths = image_filepaths(color_semantic_segmentation_directory)
    print(f"color_semantic_segmentation_filepaths = {color_semantic_segmentation_filepaths}")

    for filepath in color_semantic_segmentation_filepaths:
        color_semantic_segmentation_img = cv2.imread(filepath)
        # Convert the image to rgb
        color_semantic_segmentation_img = cv2.cvtColor(color_semantic_segmentation_img, cv2.COLOR_BGR2RGB)
        semantic_segmentation_img = np.zeros(color_semantic_segmentation_img.shape[0: 2]) #np.vectorize(apply_mapping)(color_semantic_segmentation_img, color_to_categoryIndex)
        for y in range(color_semantic_segmentation_img.shape[0]):
            for x in range(color_semantic_segmentation_img.shape[1]):
                color = tuple(color_semantic_segmentation_img[y, x])
                semantic_segmentation_img[y, x] = color_to_categoryIndex[color]
        # Save the semantic segmentation tensor
        semantic_segmentation_tsr = torch.from_numpy(semantic_segmentation_img).long()
        semantic_segmentation_tsr_filepath = os.path.join(output_directory, os.path.basename(filepath)[: -4] + ".pth")
        torch.save(semantic_segmentation_tsr, semantic_segmentation_tsr_filepath)
        # Save the image (multiplied by 50 to emphasize the objects)
        semantic_segmentation_img_filepath = os.path.join(output_directory, os.path.basename(filepath))
        cv2.imwrite(semantic_segmentation_img_filepath, 50 * semantic_segmentation_img)

def image_filepaths(images_directory):
    image_filepaths_in_directory = [os.path.join(images_directory, filename) for filename in os.listdir(images_directory)
                              if os.path.isfile(os.path.join(images_directory, filename))
                              and filename.upper().endswith('.PNG')]
    return image_filepaths_in_directory

#def apply_mapping(color, color_to_index):
#    return color_to_index[color]  # If the color is not in the dictionary, return 0

if __name__ == '__main__':
    main()