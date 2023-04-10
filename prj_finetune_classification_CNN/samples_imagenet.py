import torch
import torchvision
import os
import matplotlib.pyplot as plt
import logging
from PIL import Image
import urllib

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main():
    logging.info(f"samples_imagenet.main()")

    """image_urls = ["https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01498041_stingray.JPEG",
                  "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01614925_bald_eagle.JPEG",
                  "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01773549_barn_spider.JPEG",
                  "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02815834_beaker.JPEG",
                  "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n04009552_projector.JPEG",
                  "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n03710193_mailbox.JPEG"]
    """
    image_urls = ["https://live.staticflickr.com/1482/26709984901_36f5f7ef26_z.jpg",
                    "https://live.staticflickr.com/8752/16673976528_a1e159c088_z.jpg",
                    "https://live.staticflickr.com/2388/32076217244_5a111c9e58_z.jpg",
                    "https://live.staticflickr.com/671/30949359254_46d6d264a9_z.jpg"]

    fig, axs = plt.subplots(2, 3)
    for row in range(2):
        for col in range(2):
            with urllib.request.urlopen(image_urls[row * 2 + col]) as url:
                image_pil = Image.open(url)
                axs[row, col].imshow(image_pil)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()