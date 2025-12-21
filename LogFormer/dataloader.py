import numpy as np
import torch
from torch.utils.data import Dataset

EMBEDDING_DIM = 768


class DataGenerator(Dataset):
    def __init__(self, x, y, window_size: int, return_mask: bool = False):
        """
        Args:
            x: np.ndarray of shape (num_windows, window_size, EMBEDDING_DIM), dtype float32
            y: np.ndarray of shape (num_windows, 2), dtype float32 (one-hot)
            window_size: number of log lines per window (e.g., 20)
            return_mask: if True, also return a padding mask (x, y, pad_mask)
                         where pad_mask is True at padded (all-zero) positions.
        """
        self.x = x
        self.y = y
        self.window_size = window_size
        self.return_mask = return_mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x_np = self.x[index]  # (window_size, EMBEDDING_DIM)
        y_np = self.y[index]  # (2,)

        x_tensor = torch.from_numpy(x_np).float()
        y_tensor = torch.from_numpy(y_np).float()

        if self.return_mask:
            pad_mask = np.all(x_np == 0, axis=1)  # True where padded (all-zero) rows
            pad_mask_tensor = torch.from_numpy(pad_mask).bool()
            return x_tensor, y_tensor, pad_mask_tensor

        return x_tensor, y_tensor
