import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = split_real_imaginary(X)
        self.y = y.values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
def split_real_imaginary(X):
    real = np.real(X)
    imaginary = np.imag(X)
    return np.concatenate([real, imaginary], axis=1)