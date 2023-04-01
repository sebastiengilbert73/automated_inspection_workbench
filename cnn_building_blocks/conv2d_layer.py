import cv2
import torch
import numpy as np
import logging
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main():
    logging.info("conv2d_layer.main()")

    output_directory = "./output_conv2d_layer"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    #image_filepath = "/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images/streets/automn_street.jpg"
    image_filepath = "/home/sebastien/Documents/Programmation/educative/AutomatedInspectionWithCV/images/electronics/rpi_back.jpg"
    #image_filepath = "./images/electronics/rpi_back.jpg"

    # Convert the image to a tensor
    original_img = cv2.imread(image_filepath)
    input_tsr = torch.from_numpy(original_img).float()/255.0  # (H, W, C)
    input_tsr = torch.moveaxis(input_tsr, 2, 0)  # (C, H, W)

    # Design the convolution kernel
    weight = torch.zeros(1, 3, 5, 5)
    bias = torch.zeros(1)
    weight[0, 0, :, 0: 2] = 90#, 120, 35]  # All green
    weight[0, 1, :, 0: 2] = 120
    weight[0, 2, :, 0: 2] = 35
    weight[0, 0, :, 2:] = 75#, 160, 160)  # The right-most column has red -> yellow
    weight[0, 1, :, 2:] = 160
    weight[0, 2, :, 2:] = 160
    #weight[0, :, :, 2] = 0  # Black line in the center

    cv2.imwrite(os.path.join(output_directory, "kernel_weight.png"),  torch.moveaxis(weight.squeeze(0), 0, 2).int().numpy())
    weight = (weight - torch.mean(weight))/torch.std(weight)
    bias[0] = 0.

    # Create a 2D convolution layer with the designed weight and bias
    conv = torch.nn.Conv2d(3, 1, kernel_size=(3, 3), padding='same')
    # The tensors must be wrapped in torch.nn.Parameter objects
    conv.weight = torch.nn.Parameter(weight)
    conv.bias = torch.nn.Parameter(bias)

    # Compute the convolution on the input tensor
    convolution_tsr = conv(input_tsr)
    logging.info(f"convolution_tsr.shape = {convolution_tsr.shape}")
    logging.info(f"torch.max(convolution_tsr) = {torch.max(convolution_tsr)}; torch.min(convolution_tsr) = {torch.min(convolution_tsr)}")

    # Save the convolution image
    convolution_img = convolution_tsr.squeeze(0).detach().numpy()  # (H, W)
    cv2.imwrite(os.path.join(output_directory, "convolution.png"), 127 + 10 * convolution_img)

if __name__ == '__main__':
    main()