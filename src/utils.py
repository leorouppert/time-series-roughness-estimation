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

    print(
        f"Train segments: {len(X_train)} | Val segments: {len(X_val)} | Test segments: {len(X_test)}"
    )

    return train_loader, val_loader, test_loader


def create_sliding_windows(Z, H, window_size=25, stride=1):
    """
    Turns trajectories into sliding window segments.
    Z: (n_traj, n_points) -> X: (n_segments, window_size, 1)
    """
    Z_tensor = torch.as_tensor(Z, dtype=torch.float32)
    H_tensor = torch.as_tensor(H, dtype=torch.float32)

    windows = Z_tensor.unfold(dimension=1, size=window_size, step=stride)

    n_traj, num_windows, _ = windows.shape

    X_windows = windows.contiguous().view(-1, window_size)

    y_windows = H_tensor.repeat_interleave(num_windows)

    return X_windows, y_windows


def split_and_load_sliding_windows(
    Z,
    H,
    window_size=25,
    stride=1,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    batch_size=32,
):
    """
    Split data into train/val/test, create sliding windows (for LSTM) and return DataLoaders.

    Args:
        Z: Array of trajectories.
        H: Array of labels.
        window_size: Size of sliding window.
        stride: Stride of sliding window.
        test_size: Proportion for test.
        val_size: Proportion for validation.
        random_state: Seed for reproducibility.
        batch_size: Batch size for DataLoaders.

    Returns:
        Tuple of DataLoaders: (train_loader, val_loader, test_loader).
    """
    indices = np.arange(len(H))

    idx_train_val, idx_test = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=val_size / (1 - test_size), random_state=random_state
    )

    X_train, y_train = create_sliding_windows(
        Z[idx_train], H[idx_train], window_size, stride
    )
    X_val, y_val = create_sliding_windows(Z[idx_val], H[idx_val], window_size, stride)
    X_test, y_test = create_sliding_windows(
        Z[idx_test], H[idx_test], window_size, stride
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    print(
        f"Train segments: {len(X_train)} | Val segments: {len(X_val)} | Test segments: {len(X_test)}"
    )

    torch.cuda.empty_cache()

    return train_loader, val_loader, test_loader
