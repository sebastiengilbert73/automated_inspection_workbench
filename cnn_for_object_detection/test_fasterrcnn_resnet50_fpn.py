import torch
import torchvision
import urllib.request
from PIL import Image
import os
import copy
import numpy as np
import cv2

def main():
    print("test_fasterrcnn_resnet50_fpn.main()")

    output_directory = "./output_test_fasterrcnn_resnet50_fpn"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load a pre-trained FasterRCNN neural net
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights, box_score_thresh=0.9)
    fasterrcnn.eval()

    transform = weights.transforms()
    categories = weights.meta["categories"]

    image_url_list = [#"https://live.staticflickr.com/65535/48669724832_9f30ae364a.jpg",
                      #"https://live.staticflickr.com/7495/16327439931_2675e6b65a.jpg",
                      "https://live.staticflickr.com/1884/43570238324_5b6c54bef3.jpg"]
    annotated_imgs = []
    batch_list = []

    for img_ndx in range(len(image_url_list)):
        with urllib.request.urlopen(image_url_list[img_ndx]) as url:
            image_pil = Image.open(url)
            image_arr = np.array(image_pil)
            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
            annotated_imgs.append(image_arr)
            #image_pil.save(os.path.join(output_directory, f"image{img_ndx}.png"))
            batch_list.append(transform(image_pil))
            #prediction = fasterrcnn(batch)
            #print(f"prediction = {prediction}")
            #result_dict = prediction[0]

    batch_results = fasterrcnn(batch_list)
    print(f"batch_results: \n{batch_results}")

    for img_ndx in range(len(batch_results)):
        labels = [categories[i] for i in batch_results[img_ndx]["labels"]]
        number_of_boxes = len(labels)
        annotated_img = annotated_imgs[img_ndx]
        for box_ndx in range(number_of_boxes):
            roi_x1y1_x2y2 = batch_results[img_ndx]["boxes"][box_ndx].int().numpy()
            p1 = (roi_x1y1_x2y2[0], roi_x1y1_x2y2[1])
            p2 = (roi_x1y1_x2y2[2], roi_x1y1_x2y2[3])
            color = random_color(box_ndx)
            cv2.rectangle(annotated_img, p1, p2, color=color, thickness=2)
            cv2.putText(annotated_img, labels[box_ndx], p1, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color=(0, 0, 0), thickness=3)
            cv2.putText(annotated_img, labels[box_ndx], p1, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color=color, thickness=1)
            cv2.imwrite(os.path.join(output_directory, f"annotated_{img_ndx}.png"), annotated_img)

def random_color(index):
    color = ((index * 371)%256, (index * 1169)%256, (index * 947)%256)
    if color == (0, 0, 0):
        color = (200, 0, 0)
    return color

if __name__ == '__main__':
    main()