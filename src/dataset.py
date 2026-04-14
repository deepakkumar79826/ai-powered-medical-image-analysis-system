from pathlib import Path
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def verify_dataset_structure(train_dir: Path, val_dir: Path, test_dir: Path) -> None:
    required_dirs = [train_dir, val_dir, test_dir]
    for directory in required_dirs:
        if not directory.exists():
            raise FileNotFoundError(
                f"Missing directory: {directory}\n"
                "Expected dataset structure:\n"
                "data/raw/chest_xray/train/NORMAL\n"
                "data/raw/chest_xray/train/PNEUMONIA\n"
                "data/raw/chest_xray/val/NORMAL\n"
                "data/raw/chest_xray/val/PNEUMONIA\n"
                "data/raw/chest_xray/test/NORMAL\n"
                "data/raw/chest_xray/test/PNEUMONIA"
            )


def load_split(split_dir: Path, img_size=(224, 224), batch_size=16, shuffle=True):
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels='inferred',
        label_mode='binary',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=shuffle,
        seed=42,
    )
    return ds


def prepare_dataset(ds, training=False):
    if training:
        ds = ds.shuffle(1000)
    return ds.prefetch(buffer_size=AUTOTUNE)


def get_datasets(train_dir: Path, val_dir: Path, test_dir: Path, img_size=(224, 224), batch_size=16):
    verify_dataset_structure(train_dir, val_dir, test_dir)

    train_ds = load_split(train_dir, img_size=img_size, batch_size=batch_size, shuffle=True)
    val_ds = load_split(val_dir, img_size=img_size, batch_size=batch_size, shuffle=False)
    test_ds = load_split(test_dir, img_size=img_size, batch_size=batch_size, shuffle=False)

    class_names = train_ds.class_names

    train_ds = prepare_dataset(train_ds, training=True)
    val_ds = prepare_dataset(val_ds, training=False)
    test_ds = prepare_dataset(test_ds, training=False)

    return train_ds, val_ds, test_ds, class_names


def get_class_weights(train_ds):
    labels = []
    for _, y in train_ds.unbatch():
        labels.append(int(y.numpy()[0]))

    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()

    class_weights = {}
    for cls, count in zip(unique, counts):
        class_weights[int(cls)] = total / (len(unique) * count)

    return class_weights
