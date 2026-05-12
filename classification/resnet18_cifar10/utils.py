from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


# -------- TRAINING AND EVALUATION FUNCTIONS --------
def check_accuracy(loader, model, device):
    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
        loader: DataLoader
            The DataLoader for the dataset to check accuracy on.
        model: nn.Module
            The neural network model.
    """
    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass: compute the model output
            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        # print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")

    model.train()  # Set the model back to training mode
    return accuracy

def train_loop(dataloader, model, criterion, optimiser, iterations, device, scheduler, trainloader, testloader):
  train_losses = []
  train_accuracies = []
  test_accuracies = []
  loss_iterations = []
  accuracy_iterations = []
  current_loss_sum = 0.0
  loss_log_interval = 100 # Log average loss every this many iterations
  accuracy_log_interval = 1000 # Log accuracy every this many iterations

  # Create an iterator for the dataloader to continuously fetch batches
  data_iterator = iter(dataloader)

  for i in tqdm(range(iterations), desc="Training Iterations"):
    try:
      data, targets = next(data_iterator)
    except StopIteration:
      # Dataloader exhausted, reset iterator to cycle through data again
      data_iterator = iter(dataloader)
      data, targets = next(data_iterator)

    data = data.to(device)
    targets = targets.to(device)

    # Forward Pass: Compute model outputs
    scores = model(data)
    loss = criterion(scores, targets)

    # Backward Pass: compute the gradients
    optimiser.zero_grad()
    loss.backward()

    # Update model parameters
    optimiser.step()
    scheduler.step()

    current_loss_sum += loss.item()

    # Log average loss at specified intervals
    if (i + 1) % loss_log_interval == 0:
      avg_loss = current_loss_sum / loss_log_interval
      train_losses.append(avg_loss)
      loss_iterations.append(i + 1)
      print(f"Iteration {i+1}/{iterations} - Avg. Loss over last {loss_log_interval} iterations: {avg_loss:.4f}")
      current_loss_sum = 0.0 # Reset sum for the next interval

    # Log accuracies at specified intervals or at the very end
    if (i + 1) % accuracy_log_interval == 0 or (i + 1) == iterations:
      train_acc = check_accuracy(trainloader, model, device)
      test_acc = check_accuracy(testloader, model, device)
      train_accuracies.append(train_acc)
      test_accuracies.append(test_acc)
      accuracy_iterations.append(i + 1)
      print(f"Iteration {i+1}/{iterations} - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}%")

  return train_losses, train_accuracies, test_accuracies, loss_iterations, accuracy_iterations

# --------- PLOTTING FUNCTIONS ---------

def plot_metrics(train_losses, train_accuracies, test_accuracies, loss_iterations, accuracy_iterations, output_dir="./figures"):
    """
    Plots training loss, training error, and test error against iterations.

    Parameters:
        train_losses (list): List of average training losses logged.
        train_accuracies (list): List of training accuracies logged.
        test_accuracies (list): List of test accuracies logged.
        loss_iterations (list): Iteration numbers where losses were logged.
        accuracy_iterations (list): Iteration numbers where accuracies were logged.
    """

    # Calculate errors from accuracies
    train_errors = [100 - acc for acc in train_accuracies]
    test_errors = [100 - acc for acc in test_accuracies]

    # Plot Training Loss vs. Iterations
    plt.figure(figsize=(10, 6))
    plt.plot(loss_iterations, train_losses, label='Training Loss', color='blue')
    plt.title('Training Loss vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "training_loss.png") # Save the figure

    # Plot Training Error vs. Iterations
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_iterations, train_errors, label='Training Error', color='green')
    plt.title('Training Error vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Error (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "training_error.png") # Save the figure

    # Plot Test Error vs. Iterations
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_iterations, test_errors, label='Test Error', color='red')
    plt.title('Test Error vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Error (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "test_error.png") # Save the figure
