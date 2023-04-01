import cv2
import torch

def main():
    print("dropout2d.py")

    input_tsr = torch.randn(1, 6, 2, 2)  # (N, C, H, W)
    print(f"input_tsr = {input_tsr}")
    dropout2d = torch.nn.Dropout2d(p=0.5)
    print(f"dropout2d.training = {dropout2d.training}")
    output_tsr = dropout2d(input_tsr)
    print(f"output_tsr = {output_tsr}")

    print("training = False:")
    dropout2d.training = False
    output_tsr = dropout2d(input_tsr)
    print(f"output_tsr = {output_tsr}")




if __name__ == '__main__':
    main()