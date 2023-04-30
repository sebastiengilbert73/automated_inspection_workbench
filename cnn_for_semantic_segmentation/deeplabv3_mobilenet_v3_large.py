import torch
import torchvision
import os
import urllib
from PIL import Image
import numpy as np
import cv2

def main():
    print("deeplabv3_mobilenet_v3_large.main()")
    output_directory = "./output_deeplabv3_mobilenet_v3_large"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    weights = torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    deeplabv3 = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights,
                                                                             progress=False)
    deeplabv3.eval()

    transform = weights.transforms()
    print(f"transform = {transform}")
    categories = weights.meta["categories"]
    print(f"categories = {categories}")
    index_to_color = {0: (0, 0, 0),
                      1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
                      4: (255, 255, 0), 5: (255, 0, 255), 6: (255, 255, 0),
                      7: (64, 128, 255), 8: (64, 255, 128), 9: (128, 64, 255),
                      10: (128, 255, 64), 11: (255, 64, 128), 12: (255, 128, 64),
                      13: (32, 110, 170), 14: (32, 170, 110), 15: (110, 32, 170),
                      16: (110, 170, 32), 17: (170, 32, 110), 18: (170, 110, 32),
                      19: (100, 140, 200), 20: (100, 200, 140)}

    #image_url_list = #["https://live.staticflickr.com/909/40459189370_aaa8641626_z.jpg"]
    #image_url_list = ["https://live.staticflickr.com/142/327260700_ae77aa16cb_z.jpg"]
    image_url_list = ["https://live.staticflickr.com/65535/48669724832_9f30ae364a_z.jpg"]#,
    # "https://live.staticflickr.com/7495/16327439931_2675e6b65a_z.jpg",
    # "https://live.staticflickr.com/1884/43570238324_5b6c54bef3_z.jpg"]
    annotated_imgs = []


    for img_ndx in range(len(image_url_list)):
        with urllib.request.urlopen(image_url_list[img_ndx]) as url:
            image_pil = Image.open(url)
            image_arr = np.array(image_pil)
            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
            annotated_imgs.append(image_arr)
            original_img_filepath = os.path.join(output_directory, "original.png")
            cv2.imwrite(original_img_filepath, image_arr)
            img_tsr = transform(image_pil)
            batch_tsr = torch.zeros(1, 3, img_tsr.shape[1], img_tsr.shape[2])
            batch_tsr[0, :, :, :] = img_tsr
            print(f"img_tsr.shape = {img_tsr.shape}")
            output = deeplabv3(batch_tsr)
            print(f"type(output) = {type(output)}")
            print(f"output.keys() = {output.keys()}")
            print(f"type(output['out']) = {type(output['out'])}")
            print(f"output['out'].shape = {output['out'].shape}")
            print(f"output['aux'].shape = {output['aux'].shape}")
            semantic_segmentation_shapeNCHW = output['out'].shape
            semantic_segmentation_img = np.zeros((semantic_segmentation_shapeNCHW[2], semantic_segmentation_shapeNCHW[3], 3), dtype=np.uint8)
            #print(f"output['out'] = {output['out']}")
            semantic_segmentation_index_tsr = torch.argmax(output['out'][0, :, :, :], dim=0)

            for y in range(semantic_segmentation_img.shape[0]):
                for x in range(semantic_segmentation_img.shape[1]):
                    index = semantic_segmentation_index_tsr[y, x].item()
                    #print(f"index = {index}; type(index) = {type(index)}")
                    color = index_to_color[index]

                    semantic_segmentation_img[y, x, [0, 1, 2]] = color
            # Reshape to the original size
            semantic_segmentation_img = cv2.resize(semantic_segmentation_img, (image_arr.shape[1], image_arr.shape[0]),
                                                   interpolation=cv2.INTER_NEAREST)
            semantic_segmentation_img_filepath = os.path.join(output_directory, "semantic_segmentation.png")
            cv2.imwrite(semantic_segmentation_img_filepath, semantic_segmentation_img)

            
    #batch_results = deeplabv3(batch_list)
    #print(f"batch_results: \n{batch_results}")




if __name__ == '__main__':
    main()