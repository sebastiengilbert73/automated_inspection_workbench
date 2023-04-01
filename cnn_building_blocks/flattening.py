import torch

def main():
    print("flattening.main()")

    input_tsr = torch.randn(3, 1, 2, 2)
    print(f"input_tsr = {input_tsr}")
    print(f"input_tsr.shape = {input_tsr.shape}")
    output_tsr = input_tsr.view(-1, 1 * 2 * 2)
    print(f"output_tsr = {output_tsr}")
    print(f"output_tsr.shape = {output_tsr.shape}")


if __name__ == '__main__':
    main()