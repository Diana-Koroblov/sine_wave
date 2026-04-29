import numpy as np

def to_one_hot(index: int, total_classes: int) -> np.ndarray:
    """
    Creates a one-hot encoded vector.

    Args:
        index: The index to set to 1.
        total_classes: Total number of classes (length of the vector).

    Returns:
        A 1D numpy array of shape (total_classes,) with a 1 at the specified index.
    """
    vec = np.zeros(total_classes, dtype=np.float32)
    vec[index] = 1.0
    return vec

def sliding_window_split(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Splits a 1D array into overlapping windows.

    Args:
        data: The 1D input signal.
        window_size: The number of samples per window.

    Returns:
        A 2D numpy array of shape (num_windows, window_size), where
        num_windows = len(data) - window_size + 1.
    """
    if len(data) < window_size:
        return np.array([], dtype=np.float32).reshape(0, window_size)
    
    num_windows = len(data) - window_size + 1
    # Use stride_tricks for memory efficiency or a simple list comprehension
    # For a 10s signal at 1000Hz, simple list comprehension is fine.
    windows = [data[i : i + window_size] for i in range(num_windows)]
    return np.array(windows, dtype=np.float32)
