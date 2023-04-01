import torch

def main():
    print("batch_norm.main()")

    input_tsr = torch.randn(3, 1, 2, 2)
    print(f"input_tsr = {input_tsr}")
    batch_norm = torch.nn.BatchNorm2d(num_features=1)
    #batch_norm.training=False
    output_tsr = batch_norm(input_tsr)
    print(f"output_tsr = {output_tsr}")

if __name__ == '__main__':
    main()