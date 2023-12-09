import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Get the number of CPU cores for parallel data loading
NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_path, test_path, transform, batch_size, num_workers):
    """
    Create DataLoader instances for training and testing datasets.

    Parameters:
        - train_path (str): Path to the training dataset.
        - test_path (str): Path to the testing dataset.
        - transform (torchvision.transforms.Compose): Image transformation to be applied.
        - batch_size (int): Number of samples in each batch.
        - num_workers (int): Number of worker processes for data loading.

    Returns:
        - train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        - test_dataloader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
        - class_names (list): List of class names in the dataset.
    """

    # Create ImageFolder datasets for training and testing
    train_data = datasets.ImageFolder(train_path, transform=transform)
    test_data = datasets.ImageFolder(test_path, transform=transform)

    # Get the class names from the training dataset
    class_names = train_data.classes

    # Create DataLoader instances for training and testing datasets
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the training data for better learning
        num_workers=num_workers,
        pin_memory=True,  # Use pinned memory for faster GPU data transfer
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle the testing data
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names
