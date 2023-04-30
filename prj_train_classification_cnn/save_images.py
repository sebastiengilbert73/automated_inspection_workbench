import os
import torch
import torchvision
import logging
import random
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main():
    logging.info("save_images.main()")

    number_of_images = 12
    output_directory = "./output_save_images"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create an array of images
    number_of_rows = 3
    number_of_columns = 4
    fig, axs = plt.subplots(3, 4)
    # Load the dataset
    transform = torchvision.transforms.ToTensor()
    mnist_dataset = torchvision.datasets.MNIST("./", train=True, download=False, transform=transform)
    for row in range(number_of_rows):
        for col in range(number_of_columns):
            index = random.randint(5, len(mnist_dataset) - 1)
            img_tsr, class_ndx = mnist_dataset[index]
            axs[row, col].imshow(img_tsr.squeeze(0).numpy())
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()