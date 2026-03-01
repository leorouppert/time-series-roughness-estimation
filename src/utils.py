import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def split_and_load_data(
    Z, H, test_size=0.2, val_size=0.1, random_state=42, batch_size=32
):
    """
    Split data into train/val/test and return DataLoaders.

    Args:
        Z: Array of trajectories.
        H: Array of labels.
        test_size: Proportion for test.
        val_size: Proportion for validation.
        random_state: Seed for reproducibility.
        batch_size: Batch size for DataLoaders.

    Returns:
        Tuple of DataLoaders: (train_loader, val_loader, test_loader).
    """
    indices = np.arange(len(H))
    np.random.shuffle(indices)

    Z_shuffled = Z[indices]
    H_shuffled = H[indices]

    Z_tensor = torch.tensor(Z_shuffled, dtype=torch.float32)
    H_tensor = torch.tensor(H_shuffled, dtype=torch.float32)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        Z_tensor, H_tensor, test_size=test_size, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
    )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
