import tqdm

def train_loop(dataloader, model, criterion, optimiser, iterations, device, scheduler, trainloader, testloader):
  train_losses = []
  train_accuracies = []
  test_accuracies = []
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
      print(f"Iteration {i+1}/{iterations} - Avg. Loss over last {loss_log_interval} iterations: {avg_loss:.4f}")
      current_loss_sum = 0.0 # Reset sum for the next interval

    # Log accuracies at specified intervals or at the very end
    if (i + 1) % accuracy_log_interval == 0 or (i + 1) == iterations:
      train_acc = check_accuracy(trainloader, model)
      test_acc = check_accuracy(testloader, model)
      train_accuracies.append(train_acc)
      test_accuracies.append(test_acc)
      print(f"Iteration {i+1}/{iterations} - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}%")

  return train_losses, train_accuracies, test_accuracies