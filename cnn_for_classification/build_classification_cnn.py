import torch
import cv2

class Classif10(torch.nn.Module):
    def __init__(self, number_of_conv_kernels1,
                 number_of_conv_kernels2,
                 number_of_conv_kernels3,
                 hidden_layer_size,
                 dropout_proportion):
        super(Classif10, self).__init__()
        self.number_of_conv_kernels1 = number_of_conv_kernels1
        self.number_of_conv_kernels2 = number_of_conv_kernels2
        self.number_of_conv_kernels3 = number_of_conv_kernels3
        self.hidden_layer_size = hidden_layer_size

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=self.number_of_conv_kernels1,
                                     kernel_size=(3, 3), stride=1, padding='same')
        self.conv2 = torch.nn.Conv2d(in_channels=self.number_of_conv_kernels1, out_channels=self.number_of_conv_kernels2,
                                     kernel_size=(3, 3), stride=1, padding='same')
        self.conv3 = torch.nn.Conv2d(in_channels=self.number_of_conv_kernels2, out_channels=self.number_of_conv_kernels3,
                                     kernel_size=(3, 3), stride=1, padding='same')
        self.dropout2d = torch.nn.Dropout2d(p=dropout_proportion)
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.batchnorm2d = torch.nn.BatchNorm2d(num_features=self.number_of_conv_kernels2)
        self.linear1 = torch.nn.Linear(in_features=self.number_of_conv_kernels3 * 3 * 3, out_features=self.hidden_layer_size)
        self.linear2 = torch.nn.Linear(in_features=self.hidden_layer_size, out_features=10)
        self.dropout1d = torch.nn.Dropout1d(p=dropout_proportion)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 1, 28, 28)
        act1 = torch.nn.functional.relu(self.conv1(input_tsr))  # (N, C1, 28, 28)
        act2 = self.maxpool2d(act1)  # (N, C1, 14, 14)
        act3 = self.dropout2d(act2)  # (N, C1, 14, 14)
        act4 = torch.nn.functional.relu(self.conv2(act3))  # (N, C2, 14, 14)
        act5 = self.maxpool2d(act4)  # (N, C2, 7, 7)
        act6 = self.batchnorm2d(act5)  # (N, C2, 7, 7)
        act7 = torch.nn.functional.relu(self.conv3(act6))  # (N, C3, 7, 7)
        act8 = self.maxpool2d(act7)  # (N, C3, 3, 3)
        act9 = act8.view(-1, self.number_of_conv_kernels3 * 3 * 3)  # (N, C3 * 3 * 3)
        act10 = torch.nn.functional.relu(self.linear1(act9))  # (N, H)
        act11 = self.dropout1d(act10)  # (N, H)
        act12 = self.linear2(act11)  # (N, 10)
        return act12

def main():
    print("build_classification_cnn.main()")

    neural_net = Classif10(
        number_of_conv_kernels1=16,
        number_of_conv_kernels2=32,
        number_of_conv_kernels3=64,
        hidden_layer_size=64,
        dropout_proportion=0.5
    )

    input_tsr = torch.randn(16, 1, 28, 28)
    output_tsr = neural_net(input_tsr)
    print(f"output_tsr.shape = {output_tsr.shape}")
    print(f"output_tsr[0: 4] =\n{output_tsr[0: 4]}")

    probabilities_tsr = torch.softmax(output_tsr, dim=1)
    print(f"probabilities_tsr[0: 4] =\n{probabilities_tsr[0: 4]}")
    print(torch.sum(probabilities_tsr, dim=1))

if __name__ == '__main__':
    main()