import torch
import numpy as np
import logging
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

class TwoLayersPerceptron(torch.nn.Module):
    def __init__(self, number_of_inputs=2, hidden_layer_size=3, number_of_outputs=2,
                 dropout_ratio=0.5):
        super(TwoLayersPerceptron, self).__init__()
        self.number_of_inputs = number_of_inputs
        self.hidden_layer_size = hidden_layer_size
        self.number_of_outputs = number_of_outputs
        self.dropout_ratio = dropout_ratio

        self.linear1 = torch.nn.Linear(self.number_of_inputs, self.hidden_layer_size)
        self.linear2 = torch.nn.Linear(self.hidden_layer_size, self.number_of_outputs)
        self.dropout = torch.nn.Dropout(p=self.dropout_ratio)

    def forward(self, input_tsr):  # input_tsr.shape = (N, N_in):
        act1 = self.linear1(input_tsr)  # (N, H)
        act2 = torch.nn.functional.relu(act1)  # (N, H)
        act3 = self.dropout(act2)  # (N, H)
        act4 = self.linear2(act3)  # (N, N_out)
        return act4

class FeaturesAndClass(Dataset):
    def __init__(self, dataset_filepath):
        super(FeaturesAndClass, self).__init__()
        self.dataset_df = pd.read_csv(dataset_filepath)

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        observation = self.dataset_df.iloc[idx]  # Retrieve the observation
        features = list(observation[0: 2])  # List of features
        input_tsr = torch.tensor(features)  # Tensor of features
        class_tensor = torch.tensor(int(observation['class']))  # A tensor containing either 0 or 1
        return input_tsr, class_tensor  # Returns the input tensor and the target class index, as a tuple

def main():
    logging.info("train_neural_network.main()")

    output_directory = "./output_train_neural_network"
    dataset_filepath = "./output_generate_2D_points/dataset.csv"
    hidden_layer_size = 3
    number_of_epochs = 10
    learning_rate = 0.001
    weight_decay = 0.00001
    batch_size = 16

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the dataset
    dataset = FeaturesAndClass(dataset_filepath)
    number_of_validation_observations = round(0.2 * len(dataset))
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [len(dataset) - number_of_validation_observations, number_of_validation_observations])
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Create the neural network
    neural_net = TwoLayersPerceptron(2, hidden_layer_size, 2)

    # Training parameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(1, number_of_epochs + 1):
        print(f"Epoch {epoch}")
        # Set the neural network to training mode
        neural_net.train()
        running_loss = 0.0
        number_of_batches = 0
        for input_tsr, target_class_tsr in train_dataloader:
            # Set the parameter gradients to zero before every batch
            neural_net.zero_grad()
            # Pass the input tensor through the neural network
            output_tsr = neural_net(input_tsr)
            # Compute the loss, i.e., the error function we want to minimize
            loss = criterion(output_tsr, target_class_tsr)
            # Retropropagate the loss function, to compute the gradient of the loss function with
            # respect to every trainable parameter in the neural network
            loss.backward()
            # Perturb every trainable parameter by a small quantity, in the direction of the steepest loss descent
            optimizer.step()

            running_loss += loss.item()
            number_of_batches += 1
        average_training_loss = running_loss/number_of_batches
        print(f"average_training_loss = {average_training_loss}")



if __name__ == '__main__':
    main()