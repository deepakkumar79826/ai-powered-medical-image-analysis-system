import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from src.config import (
    TEST_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    FINAL_MODEL_PATH,
    CLASS_NAMES_PATH,
    METRICS_JSON_PATH,
    CONFUSION_MATRIX_PATH,
    SAMPLE_PREDICTIONS_PATH,
    THRESHOLD,
)
from src.dataset import load_split, prepare_dataset
from src.utils import ensure_directories, load_class_names, save_json, collect_predictions, plot_sample_predictions
from src.config import OUTPUTS_DIR, PLOTS_DIR


def evaluate_model():
    ensure_directories([OUTPUTS_DIR, PLOTS_DIR])

    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_PATH)

    test_ds = load_split(TEST_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    raw_test_ds = test_ds
    test_ds = prepare_dataset(test_ds, training=False)

    y_true, y_prob, y_pred = collect_predictions(model, test_ds, threshold=THRESHOLD)

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
    }
    save_json(metrics, METRICS_JSON_PATH)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200, bbox_inches='tight')
    plt.close()

    sample_images = []
    sample_true = []
    sample_pred = []
    sample_prob = []
    for images, labels in raw_test_ds.take(1):
        probs = model.predict(images, verbose=0).ravel()
        preds = (probs >= THRESHOLD).astype(int)
        sample_images = images.numpy()
        sample_true = labels.numpy().ravel().astype(int)
        sample_pred = preds
        sample_prob = probs
        break

    plot_sample_predictions(sample_images, sample_true, sample_pred, sample_prob, class_names, SAMPLE_PREDICTIONS_PATH)

    print('Evaluation complete.')
    print(f'Metrics saved at: {METRICS_JSON_PATH}')
    print(f'Confusion matrix saved at: {CONFUSION_MATRIX_PATH}')
    print(f'Sample predictions saved at: {SAMPLE_PREDICTIONS_PATH}')
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=class_names))


if __name__ == '__main__':
    evaluate_model()
