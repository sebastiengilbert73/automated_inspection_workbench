import torch
import numpy as np
import logging
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s [%(levelname)s] %(message)s')

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
    hidden_layer_size = 40
    dropout_ratio = 0.5
    number_of_epochs = 50
    learning_rate = 0.001
    weight_decay = 0.000001
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
    neural_net = TwoLayersPerceptron(
        number_of_inputs=2,
        hidden_layer_size=hidden_layer_size,
        number_of_outputs=2,
        dropout_ratio=dropout_ratio)

    # Training parameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Record statistics
    epochs = []
    train_losses = []
    validation_losses = []
    accuracies = []
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
        # Evaluate with the validation dataset
        # Set the neural network to evaluation (inference) mode
        neural_net.eval()
        validation_running_loss = 0.0
        number_of_batches = 0
        number_of_correct_predictions = 0
        number_of_predictions = 0
        for validation_input_tsr, validation_target_output_tsr in validation_dataloader:
            # Pass the input tensor through the neural network
            validation_output_tsr = neural_net(validation_input_tsr)
            # Compute the validation loss
            validation_loss = criterion(validation_output_tsr, validation_target_output_tsr)
            validation_running_loss += validation_loss.item()
            number_of_correct_predictions += numberOfCorrectPredictions(validation_output_tsr, validation_target_output_tsr)
            number_of_predictions += validation_input_tsr.shape[0]
            number_of_batches += 1
        average_validation_loss = validation_running_loss/number_of_batches
        accuracy = number_of_correct_predictions/number_of_predictions
        print(f"average_training_loss = {average_training_loss}; average_validation_loss = {average_validation_loss}; accuracy = {accuracy}")
        epochs.append(epoch)
        train_losses.append(average_training_loss)
        validation_losses.append(average_validation_loss)
        accuracies.append(accuracy)

    # Display the metrics evolution
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.plot(epochs, train_losses, color='b', label='Training loss')
    ax1.plot(epochs, validation_losses, color='r', label='Validation loss')
    ax1.grid(True)
    ax1.legend(loc='right')
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('accuracy', color='g')
    ax2.plot(epochs, accuracies, color='g', label='Accuracy')
    ax2.legend(loc='upper right')

    # Display the decision boundary
    neural_net.eval()  # Set the neural network to evaluation (inference) mode
    delta_v = 0.1
    predicted_0_xs = []
    predicted_0_ys = []
    predicted_1_xs = []
    predicted_1_ys = []
    for v0 in np.arange(-4.0, 4.1, delta_v):
        for v1 in np.arange(-4.0, 4.1, delta_v):
            input_tsr = torch.zeros(1, 2)
            input_tsr[0, 0] = v0
            input_tsr[0, 1] = v1
            output_tsr = neural_net(input_tsr)  # (1, 2)
            predicted_class = 1
            if output_tsr[0, 0] > output_tsr[0, 1]:
                predicted_class = 0
            if predicted_class == 0:
                predicted_0_xs.append(v0)
                predicted_0_ys.append(v1)
            else:
                predicted_1_xs.append(v0)
                predicted_1_ys.append(v1)

    observation_0_xs = []
    observation_0_ys = []
    observation_1_xs = []
    observation_1_ys = []
    for input_tsr, target_class_tsr in train_dataloader:
        for obs_ndx in range(input_tsr.shape[0]):
            if target_class_tsr[obs_ndx].item() == 0:
                observation_0_xs.append(input_tsr[obs_ndx, 0].item())
                observation_0_ys.append(input_tsr[obs_ndx, 1].item())
            else:
                observation_1_xs.append(input_tsr[obs_ndx, 0].item())
                observation_1_ys.append(input_tsr[obs_ndx, 1].item())

    fig, ax = plt.subplots()
    scatter0 = ax.scatter(predicted_0_xs, predicted_0_ys, c='darkorange', marker='s', s=25)
    scatter1 = ax.scatter(predicted_1_xs, predicted_1_ys, c='cornflowerblue', marker='s', s=25)
    scatter_obs0 = ax.scatter(observation_0_xs, observation_0_ys, c='r', marker='.')
    scatter_obs1 = ax.scatter(observation_1_xs, observation_1_ys, c='b', marker='.')



    plt.show()

def numberOfCorrectPredictions(predictions_tsr, target_class_tsr):
    return sum(torch.argmax(predictions_tsr, dim=1) == target_class_tsr).item()

if __name__ == '__main__':
    main()