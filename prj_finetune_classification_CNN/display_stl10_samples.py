import torch
import torchvision
import matplotlib.pyplot as plt
import random

def main():
    print("display_stl10_samples.main()")

    transform = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    stl10_dataset = torchvision.datasets.STL10(root='./', split='train', transform=transform,
                                               download=True)
    number_of_rows = 3
    number_of_columns = 3
    fig, axs = plt.subplots(number_of_rows, number_of_columns)
    for row in range(number_of_rows):
        for col in range(number_of_columns):
            index = random.randint(0, len(stl10_dataset) - 1)  # Choose a random index from the dataset
            img_tsr, class_ndx = stl10_dataset[index]  # Get the image tensor and the target class index
            img_tsr = torch.moveaxis(img_tsr, 0, 2)  # (C, H, W) -> (H, W, C)
            # Set the range to [0, 1]
            min_val = torch.min(img_tsr)
            max_val = torch.max(img_tsr)
            img_tsr = (img_tsr - min_val) / (max_val - min_val)
            axs[row, col].imshow(img_tsr.squeeze(0).numpy())
    plt.show()

if __name__ == '__main__':
    main()