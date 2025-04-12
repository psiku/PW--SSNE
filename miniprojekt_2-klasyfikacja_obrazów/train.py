import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def train(model, train_loader, criterion, optimizer, device, epochs, validation_loader, class_names, scheduler=None):
    epoch_acc = []
    epoch_loss = []

    val_acc_list = []
    val_loss_list = []

    for epoch in range(epochs):

        batch_acc = []
        running_loss = 0.0

        model.train()

        for i, data in tqdm(enumerate(train_loader, 0), desc=f"Training Epoch [{epoch+1}/{epochs}]", total=len(train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            accuracy = accuracy_score(
                labels.cpu().numpy(),
                torch.argmax(outputs, dim=1).cpu().numpy()
                )

            batch_acc.append(accuracy)

        avg_loss = running_loss / len(train_loader)
        avg_acc = sum(batch_acc) / len(batch_acc)

        print(f"Training -> Loss: {avg_loss:.4f} - Acc: {avg_acc * 100:.2f}%")

        epoch_loss.append(avg_loss)
        epoch_acc.append(avg_acc)

        model.eval()

        val_labels = []
        val_preds = []
        val_loss = 0.0
        with torch.no_grad():
            for data in tqdm(validation_loader, desc=f"Validating Epoch [{epoch+1}/{epochs}]", total=len(validation_loader)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predictions = torch.max(outputs, 1)
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(validation_loader)
        val_acc = accuracy_score(val_labels, val_preds)

        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(f"Validation -> Loss: {val_loss:.4f} - Acc: {val_acc * 100:.2f}%")

        # scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

    print("Finished Training")

    return model, epoch_loss, epoch_acc, val_loss_list, val_acc_list
