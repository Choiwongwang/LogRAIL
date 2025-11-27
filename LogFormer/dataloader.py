import numpy as np
import torch
from torch.utils.data import Dataset
import math

EMBEDDING_DIM = 768

class DataGenerator(Dataset):
    def __init__(self, x, y, window_size):
        """
        x: numpy.ndarray, shape = (num_windows, ?, EMBEDDING_DIM), dtype=np.float32
        y: numpy.ndarray, shape = (num_windows, 2), dtype=np.float32
        window_size: int (e.g., 20)
        """
        self.x = x
        self.y = y
        self.window_size = window_size

    def __len__(self):
        # number of samples = number of windows
        return len(self.x)

    def __getitem__(self, index):
        # 1) NumPy array → Tensor
        #    self.x[index] should have shape (window_size, EMBEDDING_DIM)
        x_np = self.x[index]    # (window_size, 768), dtype=float32
        y_np = self.y[index]    # (2,),              dtype=float32

        # 2) If rows < window_size (not expected), you could pad manually:
        #
        # num_rows = x_np.shape[0]
        # if num_rows < self.window_size:
        #     pad = np.zeros((self.window_size - num_rows, EMBEDDING_DIM), dtype=np.float32)
        #     x_np = np.concatenate([x_np, pad], axis=0)
        # else:
        #     x_np = x_np[: self.window_size]  # truncate if longer

        # 3) Wrap as Tensor
        x_tensor = torch.from_numpy(x_np).float()  # shape = (window_size, EMBEDDING_DIM)
        y_tensor = torch.from_numpy(y_np).float()  # shape = (2,)

        return x_tensor, y_tensor
