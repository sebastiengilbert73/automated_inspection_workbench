import torch

def main():
    print("manipulate_tensors.main()")

    # We can create tensors as multidimensional arrays, just like numpy arrays
    t1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t2 = 2 * t1
    print(t2)
    t3 = t1 + t2
    print(t3)
    print(f"t3.grad_fn = {t3.grad_fn}")
    # PyTorch offers a wide range of functions applicable to tensors
    cos_t1 = torch.cos(t1)
    print (f"cos_t1 = {cos_t1}")
    v1 = torch.tensor([[4.], [3.], [2.]])
    print (f"v1.shape = {v1.shape}")
    t1xv1 = t1 @ v1
    print(f"t1xv1 = {t1xv1}")


if __name__ == '__main__':
    main()