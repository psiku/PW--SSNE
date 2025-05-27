import torch
from torch.utils.data import Dataset
import numpy as np


def pad_or_trim(seq: np.ndarray, max_len: int, pad_value: int = 0) -> np.ndarray:
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        pad_width = max_len - len(seq)
        return np.concatenate([seq, np.full(pad_width, pad_value, dtype=seq.dtype)])


def shift_pad(seq: np.ndarray, shift: int) -> np.ndarray:
    return seq + shift


class MusicDataset(Dataset):
    def __init__(self, raw_data, boundry, pad_value, if_embedding=True, has_label=True):
        self.samples = []
        for item in raw_data:

            if has_label:
                seq, label = item
            else:
                seq = item
                label = -1
            shifted_seq = shift_pad(seq, 2)
            padded_seq = pad_or_trim(shifted_seq, boundry, pad_value)
            input_dtype = torch.long if if_embedding else torch.float32
            self.samples.append((
                torch.tensor(padded_seq, dtype=input_dtype),
                torch.tensor(label, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]