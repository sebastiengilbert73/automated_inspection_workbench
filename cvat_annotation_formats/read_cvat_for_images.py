import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch

def main():
    print("read_cvat_for_images.main()")
    output_directory = "./output_read_cvat_for_images"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    category_to_index = {'background': 0, 'capacitor': 1, 'diode': 2, 'main chip': 3, 'substrate': 4}

    annotation_filepath = "./images/microcontrollers_cvat/annotations.xml"
    annotations_tree = ET.parse(annotation_filepath)
    annotations_root_elm = annotations_tree.getroot()

    # Extract the list of 'image' elements
    image_elm_list = annotations_root_elm.findall('image')

    # Loop through the image elements
    for image_elm in image_elm_list:
        image_filename = image_elm.attrib['name']
        width = int(image_elm.attrib['width'])
        height = int(image_elm.attrib['height'])
        # Create a semantic segmentation with integers representing categories
        semantic_segmentation_img = np.zeros((height, width), dtype=np.uint8)
        polygon_elm_list = image_elm.findall('polygon')
        # Draw the substrate first, since it is behind all the other components
        for polygon_elm in polygon_elm_list:
            if polygon_elm.attrib['label'] == 'substrate':
                DrawPolygon(polygon_elm, semantic_segmentation_img, category_to_index)
        # Draw all the other objects
        for polygon_elm in polygon_elm_list:
            if polygon_elm.attrib['label'] != 'substrate':
                DrawPolygon(polygon_elm, semantic_segmentation_img, category_to_index)
        semantic_segmentation_img_filepath = os.path.join(output_directory, f"{image_filename}_segm.png")
        cv2.imwrite(semantic_segmentation_img_filepath, semantic_segmentation_img)
        # Convert to a PyTorch tensor of long
        semantic_segmentation_tsr = torch.from_numpy(semantic_segmentation_img).long()
        semantic_segmentation_tsr_filepath = os.path.join(output_directory, f"{image_filename}_segm.pth")
        torch.save(semantic_segmentation_tsr, semantic_segmentation_tsr_filepath)

def DrawPolygon(polygon_elm, semantic_segmentation_img, category_to_index):
    label = polygon_elm.attrib['label']
    points_str = polygon_elm.attrib['points']  # Ex.: '356.61,200.55;363.63,200.03;364.15,217.45;356.87,217.45'
    # Split the string on ';'
    pairs_str_list = points_str.split(';')
    # Convert each string into a pair of integer numbers
    points = []
    for pair_ndx in range(len(pairs_str_list)):
        x, y = pairs_str_list[pair_ndx].split(',')
        # Round to the nearest integer
        x = round(float(x))
        y = round(float(y))
        points.append((x, y))
    cv2.fillPoly(semantic_segmentation_img, [np.array(points)], category_to_index[label])

if __name__ == '__main__':
    main()