import torch
import torchvision
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main():
    logging.info("googlenet_for_cifar10.main()")

    output_directory = "./output_googlenet_for_cifar10"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the GoogLeNet CNN
    googlenet = torchvision.models.googlenet(weights='DEFAULT')
    transform = torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1.transforms()

    # Load the CIFAR10 dataset
    cifar10_dataset = torchvision.datasets.CIFAR10(root='./', train=True, transform=transform,
                                                   download=True)
    train_dataset, validation_dataset = torch.utils.data.random_split(cifar10_dataset, [0.8, 0.2])
    train_indices = torch.randperm(len(train_dataset))[: 1000]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    validation_indices = torch.randperm(len(validation_dataset))[0: 200]
    validation_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)
    #print(f"len(train_dataset) = {len(train_dataset)}; len(validation_dataset) = {len(validation_dataset)}")
    batch_size = 16
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler)

    # Pass a random tensor through the CNN
    input_tsr = torch.randn(16, 3, 32, 32)
    # Transform the batch tensor
    input_tsr = transform(input_tsr)
    logging.info(f"input_tsr.shape = {input_tsr.shape}")  # Upsampled to (16, 3, 224, 224)
    output_tsr = googlenet(input_tsr)
    logging.info(f"output_tsr.shape = {output_tsr.shape}")  # (16, 1000)

    # Replace the fc layer with a linear layer with 10 outputs
    googlenet.fc = torch.nn.Linear(in_features=1024, out_features=10)
    # Pass the random tensor through the CNN
    output_tsr = googlenet(input_tsr)
    logging.info(f"output_tsr.shape = {output_tsr.shape}")  # (16, 10)

    # Training parameters
    learning_rate = 0.001
    weight_decay = 0.000001
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(googlenet.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Record statistics
    epochs = []
    train_losses = []
    validation_losses = []
    accuracies = []
    number_of_epochs = 10
    for epoch in range(1, number_of_epochs + 1):
        # Set the neural network to training mode
        googlenet.train()
        running_loss = 0.0
        number_of_batches = 0
        for input_tsr, target_class_tsr in train_dataloader:
            # Set the parameter gradients to zero before every batch
            googlenet.zero_grad()
            # Pass the input tensor through the neural network
            output_tsr = googlenet(input_tsr)
            # Compute the loss, i.e., the error function we want to minimize
            loss = criterion(output_tsr, target_class_tsr)
            # Back-propagate the loss function, to compute the gradient of the loss function with
            # respect to every trainable parameter in the neural network
            loss.backward()
            # Perturb every trainable parameter by a small quantity, in the direction of the steepest loss descent
            optimizer.step()

            running_loss += loss.item()
            number_of_batches += 1
            if number_of_batches % 10 == 1:
                print('.', flush=True, end='')
        average_training_loss = running_loss / number_of_batches

        # Evaluate with the validation dataset
        # Set the neural network to evaluation (inference) mode
        googlenet.eval()
        validation_running_loss = 0.0
        number_of_batches = 0
        number_of_correct_predictions = 0
        number_of_predictions = 0
        for validation_input_tsr, validation_target_output_tsr in validation_dataloader:
            # Pass the input tensor through the neural network
            validation_output_tsr = googlenet(validation_input_tsr)
            # Compute the validation loss
            validation_loss = criterion(validation_output_tsr, validation_target_output_tsr)
            validation_running_loss += validation_loss.item()
            number_of_correct_predictions += numberOfCorrectPredictions(validation_output_tsr,
                                                                        validation_target_output_tsr)
            number_of_predictions += validation_input_tsr.shape[0]
            number_of_batches += 1
        average_validation_loss = validation_running_loss / number_of_batches
        accuracy = number_of_correct_predictions / number_of_predictions
        print(
            f"Epoch {epoch}: average_training_loss = {average_training_loss}; average_validation_loss = {average_validation_loss}; accuracy = {accuracy}")
        epochs.append(epoch)
        train_losses.append(average_training_loss)
        validation_losses.append(average_validation_loss)
        accuracies.append(accuracy)


    # Display the evolution of the metrics
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.plot(epochs, train_losses, color='b', label='Training loss')
    ax1.plot(epochs, validation_losses, color='r', label='Validation loss')
    ax1.grid(True)
    ax1.legend(loc='right')
    ax2 = ax1.twinx()  # Instantiate a second axis that shares the same x-axis
    ax2.set_ylabel('accuracy', color='g')
    ax2.plot(epochs, accuracies, color='g', label='Accuracy')
    ax2.legend(loc='upper right')
    plt.savefig("./output/0_epochLoss.png")
def numberOfCorrectPredictions(predictions_tsr, target_class_tsr):
    return sum(torch.argmax(predictions_tsr, dim=1) == target_class_tsr).item()

if __name__ == '__main__':
    main()