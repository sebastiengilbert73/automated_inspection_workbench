import ast
import torch
import torchvision
from PIL import Image
import urllib.request

def main():
    print("googlenet.main()")

    # Load a pre-trained GoogLeNet CNN
    googlenet = torchvision.models.googlenet(weights='DEFAULT')
    googlenet.eval()
    transform = torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1.transforms()

    # ImageNet classes
    imagenet_classes_filepath = "./imagenet1000_clsidx_to_labels.txt"
    imagenet_classes_content = None
    with open(imagenet_classes_filepath, 'r') as imagenet_classes_file:
        imagenet_classes_content = imagenet_classes_file.read()
    classNdx_to_name = ast.literal_eval(imagenet_classes_content)

    #img_filepath = "/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images/animals/Komodo_dragon.jpg"
    #img_filepath = "/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images/hens/cayenne2.jpg"
    #img_filepath = "/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images/fruits/bananas_black1b_noise.jpg"
    img_filepath = "/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images/cookies/graham_black.jpg"

    img_url = "https://live.staticflickr.com/1482/26709984901_36f5f7ef26_z.jpg"
    with urllib.request.urlopen(img_url) as url:
        image_pil = Image.open(url)

    #image_pil = Image.open(img_url)
    image_tsr = transform(image_pil)
    print(f"image_tsr.shape = {image_tsr.shape}")
    output_tsr = googlenet(image_tsr.unsqueeze(0))
    #print(f"output_tsr =\n{output_tsr}")

    # Predicted class
    predicted_class_ndx = torch.argmax(output_tsr, dim=1).item()
    print(f"predicted_class_ndx = {predicted_class_ndx}")
    print(f"Predicted class: {classNdx_to_name[predicted_class_ndx]}")

if __name__ == '__main__':
    main()