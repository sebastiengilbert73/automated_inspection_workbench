import torch

linear_layer = torch.nn.Linear(3, 2)
linear_parameters = list(linear_layer.named_parameters())
print(linear_parameters)

# Test the use of the linear layer
input_tsr = torch.tensor([[1., 2., 3.]])  # (1, 3)
print(f"input_tsr.shape = {input_tsr.shape}")
output_tsr = linear_layer(input_tsr)
print(f"output_tsr =\n{output_tsr}")
# Test:
output2_tsr = torch.matmul(linear_layer.weight, input_tsr[0]) + linear_layer.bias
print(f"output2_tsr =\n{output2_tsr}")