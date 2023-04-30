import torch
import torchvision
from PIL import Image
import urllib.request
import ast

def main():
    print("resnet50.main()")

    # ImageNet classes
    imagenet_classes_filepath = "./imagenet1000_clsidx_to_labels.txt"
    imagenet_classes_content = None
    with open(imagenet_classes_filepath, 'r') as imagenet_classes_file:
        imagenet_classes_content = imagenet_classes_file.read()
    classNdx_to_name = ast.literal_eval(imagenet_classes_content)

    resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
                                           progress=False)
    resnet50.eval()
    transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()

    #img_url = "https://live.staticflickr.com/1482/26709984901_36f5f7ef26_z.jpg"
    #img_url = "https://live.staticflickr.com/8752/16673976528_a1e159c088_z.jpg"
    #img_url = "https://live.staticflickr.com/2388/32076217244_5a111c9e58_z.jpg"
    img_url = "https://live.staticflickr.com/671/30949359254_46d6d264a9_z.jpg"
    with urllib.request.urlopen(img_url) as url:
        image_pil = Image.open(url)

    # image_pil = Image.open(img_url)
    image_tsr = transform(image_pil)
    print(f"image_tsr.shape = {image_tsr.shape}")
    output_tsr = resnet50(image_tsr.unsqueeze(0))

    # Predicted class
    predicted_class_ndx = torch.argmax(output_tsr, dim=1).item()
    print(f"predicted_class_ndx = {predicted_class_ndx}")
    print(f"Predicted class: {classNdx_to_name[predicted_class_ndx]}")

if __name__ == '__main__':
    main()