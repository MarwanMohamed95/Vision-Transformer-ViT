import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Perform a single training step.

    Parameters:
        - model (torch.nn.Module): The neural network model.
        - dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        - loss_fn (torch.nn.Module): The loss function.
        - optimizer (torch.optim.Optimizer): The optimizer.
        - device (torch.device): The device on which to perform the computation.

    Returns:
        Tuple[float, float]: Training loss and accuracy.
    """
    
    # Set the model to training mode
    model.train()

    # Initialize training loss and accuracy
    train_loss, train_acc = 0.0, 0.0

    # Iterate over batches in the training DataLoader
    for X, y in dataloader:
        # Move data to the specified device (GPU or CPU)
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        
        # Compute the loss
        loss = loss_fn(y_pred, y)

        # Accumulate the training loss
        train_loss += loss.item()

        # Zero out the gradients, perform backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        class_pred = torch.argmax(y_pred, dim=1)
        train_acc += (class_pred == y).sum().item() / len(y)

    # Average the training loss and accuracy
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    Perform a single testing/validation step.

    Parameters:
        - model (torch.nn.Module): The neural network model.
        - dataloader (torch.utils.data.DataLoader): DataLoader for the testing/validation dataset.
        - loss_fn (torch.nn.Module): The loss function.
        - device (torch.device): The device on which to perform the computation.

    Returns:
        Tuple[float, float]: Testing/validation loss and accuracy.
    """
    
    # Set the model to evaluation mode
    model.eval()

    # Initialize testing/validation loss and accuracy
    test_loss, test_acc = 0.0, 0.0

    # Iterate over batches in the testing/validation DataLoader
    for X, y in dataloader:
        # Move data to the specified device (GPU or CPU)
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Compute the loss
        loss = loss_fn(y_pred, y)

        # Accumulate the testing/validation loss
        test_loss += loss.item()

        # Calculate testing/validation accuracy
        class_pred = torch.argmax(y_pred, dim=1)
        test_acc += (class_pred == y).sum().item() / len(y)

    # Average the testing/validation loss and accuracy
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader, 
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """
    Train a neural network model.

    Parameters:
        - model (torch.nn.Module): The neural network model.
        - train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        - test_dataloader (torch.utils.data.DataLoader): DataLoader for the testing/validation dataset.
        - loss_fn (torch.nn.Module): The loss function.
        - optimizer (torch.optim.Optimizer): The optimizer.
        - epochs (int): Number of training epochs.
        - device (torch.device): The device on which to perform the computation.

    Returns:
        Dict[str, List]: Dictionary containing training and testing/validation results.
    """
    
    # Initialize a dictionary to store training and testing/validation results
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Move the model to the specified device (GPU or CPU)
    model.to(device)

    # Iterate over epochs
    for epoch in tqdm(range(epochs)):
        # Perform a training step
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)

        # Perform a testing/validation step
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        # Print out training and testing/validation metrics
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update the results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results
