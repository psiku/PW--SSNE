import torch
from tqdm import tqdm

def evaluate_model(model, test_dataloader, device, verbose=True):
    model.eval()
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

            for label in torch.unique(targets):
                mask = targets == label
                count = mask.sum().item()
                correct = (predictions[mask] == targets[mask]).sum().item()
                class_total[label.item()] = class_total.get(label.item(), 0) + count
                class_correct[label.item()] = class_correct.get(label.item(), 0) + correct

    acc_per_class = {cls: class_correct[cls] / class_total[cls] for cls in class_total}
    mean_acc = sum(acc_per_class.values()) / len(acc_per_class)
    if verbose:
        print("Accuracy per class (%):")
        for cls, acc in acc_per_class.items():
            print(f"Class {cls}: {acc * 100:.2f}%")
        print(f"\nMean Accuracy: {mean_acc * 100:.2f}%")

    return acc_per_class, mean_acc