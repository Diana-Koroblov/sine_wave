import numpy as np

from src.data.processor import sliding_window_split, to_one_hot


def test_to_one_hot():
    """Verify one-hot vector dimensions and values."""
    total = 4
    for i in range(total):
        vec = to_one_hot(i, total)
        assert vec.shape == (total,)
        assert vec[i] == 1.0
        assert np.sum(vec) == 1.0


def test_sliding_window_split_shape():
    """Verify the output shape of sliding windows."""
    data = np.arange(100)
    window_size = 10
    windows = sliding_window_split(data, window_size)

    # Expected: 100 - 10 + 1 = 91 windows
    assert windows.shape == (91, 10)


def test_sliding_window_split_values():
    """Verify that windows correctly slice the input data."""
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    window_size = 3
    windows = sliding_window_split(data, window_size)

    # Expected windows: [1,2,3], [2,3,4], [3,4,5]
    expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.float32)

    np.testing.assert_array_equal(windows, expected)


def test_sliding_window_split_short_data():
    """Verify behavior when data is shorter than window size."""
    data = np.array([1, 2], dtype=np.float32)
    window_size = 5
    windows = sliding_window_split(data, window_size)
    assert windows.shape == (0, 5)
