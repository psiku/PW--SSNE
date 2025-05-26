import torch
from .map_data import map_label_to_name, map_name_to_label
from ..music_dataset.MusicDataset import MusicDataset


def plot_label_distribution(dataset: MusicDataset, title: str):
    labels = [label.item() if isinstance(label, torch.Tensor) else label for _, label in dataset]


    unique_label_names = {map_label_to_name(label) for label in labels}

    counts = {name: labels.count(map_name_to_label(name)) for name in unique_label_names}

    ordered_names = sorted(unique_label_names)
    ordered_counts = [counts[name] for name in ordered_names]
    percentages = [count / len(labels) * 100 for count in ordered_counts]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.bar(ordered_names, ordered_counts, color='skyblue', edgecolor='black')
    plt.xlabel("Labels")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    for name, count, percentage in zip(ordered_names, ordered_counts, percentages):
        print(f"{name}: {count} ({percentage:.2f}%)")