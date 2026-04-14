import tensorflow as tf
from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    OUTPUTS_DIR,
    PLOTS_DIR,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    CLASS_NAMES_PATH,
    HISTORY_JSON_PATH,
    TRAINING_CURVES_PATH,
)
from src.dataset import get_datasets, get_class_weights
from src.model import build_model
from src.utils import ensure_directories, save_class_names, save_json, plot_training_curves


def train_model():
    ensure_directories([MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR])

    train_ds, val_ds, test_ds, class_names = get_datasets(
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    print(f"Class names: {class_names}")
    class_weights = get_class_weights(train_ds)
    print(f"Class weights: {class_weights}")

    model = build_model(img_size=IMG_SIZE, learning_rate=LEARNING_RATE)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save(FINAL_MODEL_PATH)
    save_class_names(class_names, CLASS_NAMES_PATH)
    save_json(history.history, HISTORY_JSON_PATH)
    plot_training_curves(history, TRAINING_CURVES_PATH)

    test_results = model.evaluate(test_ds, verbose=1)
    metrics = dict(zip(model.metrics_names, [float(x) for x in test_results]))
    save_json(metrics, OUTPUTS_DIR / 'final_test_metrics_after_train.json')

    print('\nTraining complete.')
    print(f'Best model saved at: {BEST_MODEL_PATH}')
    print(f'Final model saved at: {FINAL_MODEL_PATH}')
    print(f'Training plots saved at: {TRAINING_CURVES_PATH}')


if __name__ == '__main__':
    train_model()
