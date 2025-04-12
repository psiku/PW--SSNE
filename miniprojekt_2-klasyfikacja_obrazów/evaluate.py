import torch
from tqdm import tqdm


def evaluate(model, test_dataloader, classes, device):
    model.eval()

    all_labels = []
    all_preds = []

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Evaluating", leave=False):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())

            # batch sample
            for label, prediction in zip(labels, predictions):
                class_name = classes[label]
                total_pred[class_name] += 1
                if label == prediction:
                    correct_pred[class_name] += 1

    total_correct = sum(correct_pred.values())
    total_samples = sum(total_pred.values())
    total_acc = total_correct / total_samples if total_samples > 0 else 0.0

    print("\nPer-class Accuracy:")
    aggregated_acc = 0.0
    for classname in classes:
        if total_pred[classname] > 0:
            accuracy = correct_pred[classname] / total_pred[classname]
            print(f"Accuracy for class '{classname:15}': {accuracy:.1%}")
            aggregated_acc += accuracy
        else:
            print(f"Accuracy for class '{classname:15}': N/A (no samples)")

    avg_class_acc = aggregated_acc / len(classes)

    print(f"\nTotal accuracy:         {total_acc:.2%}")
    print(f"Average class accuracy: {avg_class_acc:.2%}")
    print("Finished Evaluation")

    return all_labels, all_preds
