import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


def training_plot(epochs, title, train_loss, train_acc, val_loss, val_acc, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_loss, label='Train Loss', color='blue')
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Training Loss" + title)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_acc, label='Train Accuracy', color='blue')
    plt.plot(range(1, epochs + 1), val_acc, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy' + title)
    plt.legend()
    plt.grid(True)

    plt.show()


def evaluation_confusion_matrix(y_true, y_pred, class_names, figsize=(15, 10)):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def class_bar_plot(y_true, y_pred, class_names, figsize=(15, 10)):
    correct = {name: 0 for name in class_names}
    incorrect = {name: 0 for name in class_names}

    for true, pred in zip(y_true, y_pred):
        class_name = class_names[true]
        if true == pred:
            correct[class_name] += 1
        else:
            incorrect[class_name] += 1

    labels = list(class_names)
    correct_vals = [correct[name] for name in labels]
    incorrect_vals = [incorrect[name] for name in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=figsize)
    plt.bar(x - width/2, correct_vals, width, label='Correct')
    plt.bar(x + width/2, incorrect_vals, width, label='Incorrect')

    plt.ylabel('Number of Samples')
    plt.title('Correct vs Incorrect Predictions per Class')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()