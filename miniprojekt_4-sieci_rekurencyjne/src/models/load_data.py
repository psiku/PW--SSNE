from sklearn.model_selection import train_test_split
from src.music_dataset.MusicDataset import MusicDataset
from collections import Counter
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Tuple, Optional, Any, List
from torch.utils.data import Dataset, DataLoader, Sampler

def split_data(data: MusicDataset, train_ratio: float = 0.7) -> Tuple[List[Any], List[Any], List[Any], List[Any], List[Any], List[Any]]:
    x_train, x_test, y_train, y_test = train_test_split(
        [seq_label[0] for seq_label in data],
        [seq_label[1] for seq_label in data],
        train_size=train_ratio,
        random_state=42,
        stratify=[seq_label[1] for seq_label in data]
    )

    x_train_final, x_val, y_train_final, y_val = train_test_split(
        x_train,
        y_train,
        train_size=train_ratio,
        random_state=42,
        stratify=y_train
    )

    return x_train_final, y_train_final, x_val, y_val, x_test, y_test


def create_dataset(x_values: List[Any], y_values: List[Any], max_len: int, pad_value: int = 0, if_embedding=True) -> MusicDataset:
    return MusicDataset(list(zip(x_values, y_values)), max_len, pad_value, if_embedding)


def get_sampler(dataset: Dataset) -> WeightedRandomSampler:
    labels = [label for _, label in dataset]
    label_counts = Counter(labels)

    num_samples = len(dataset)
    cls_weights = {label: num_samples / count for label, count in label_counts.items()}

    sample_weights = [cls_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def create_dataloader(dataset: Dataset, batch_size: int, sampler: Optional[Sampler] = None) -> DataLoader:
    if sampler is None:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return dataloader