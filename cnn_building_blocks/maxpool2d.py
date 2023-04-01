import torch

def main():
    print(f"maxpool2d.main()")

    input_tsr = torch.randn(1, 1, 4, 4)
    print(f"input_tsr = \n{input_tsr}")
    maxpool2d = torch.nn.MaxPool2d(kernel_size=(2, 2))
    output_tsr = maxpool2d(input_tsr)
    print(f"output_tsr = \n{output_tsr}")

if __name__ == '__main__':
    main()