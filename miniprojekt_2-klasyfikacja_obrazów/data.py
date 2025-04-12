import torch
import torchvision


def split_data(image_dir, transform, train_size):

    trainset = torchvision.datasets.ImageFolder(
        root=image_dir,
        transform=transform
        )

    num_train = int(len(trainset) * train_size)
    num_val = len(trainset) - num_train

    train_subset, val_subset = torch.utils.data.random_split(
        trainset, [num_train, num_val],
        generator=torch.Generator().manual_seed(42)
    )

    return train_subset, val_subset, trainset.classes


def create_dataloader(subset, shuffle, batch_size, num_workers):
    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader
