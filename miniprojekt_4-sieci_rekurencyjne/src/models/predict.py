import torch

def predict(model, dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions


def save_predictions(predictions, output_file):
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Predictions saved to {output_file}")