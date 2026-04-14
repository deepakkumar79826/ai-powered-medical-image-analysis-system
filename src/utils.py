import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def ensure_directories(paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_class_names(class_names, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")


def load_class_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def plot_training_curves(history, output_path):
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict['loss'], label='Train Loss')
    plt.plot(epochs, history_dict['val_loss'], label='Val Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history_dict['val_accuracy'], label='Val Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def collect_predictions(model, dataset, threshold=0.5):
    y_true = []
    y_prob = []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0).ravel()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().ravel().astype(int).tolist())

    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    return y_true, y_prob, y_pred


def plot_sample_predictions(images, true_labels, pred_labels, pred_probs, class_names, output_path):
    num_images = min(len(images), 9)
    plt.figure(figsize=(12, 12))

    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype('uint8'))
        plt.axis('off')
        true_name = class_names[int(true_labels[i])]
        pred_name = class_names[int(pred_labels[i])]
        confidence = pred_probs[i]
        plt.title(f"True: {true_name}\nPred: {pred_name} ({confidence:.2f})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
