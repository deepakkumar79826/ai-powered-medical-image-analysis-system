from pathlib import Path
import numpy as np
import tensorflow as tf
from src.config import FINAL_MODEL_PATH, CLASS_NAMES_PATH, IMG_SIZE, THRESHOLD
from src.utils import load_class_names


def preprocess_single_image(image_path, img_size=(224, 224)):
    image = tf.keras.utils.load_img(image_path, target_size=img_size, color_mode='rgb')
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_image(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found: {image_path}')

    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_PATH)

    image_array = preprocess_single_image(image_path, IMG_SIZE)
    probability = float(model.predict(image_array, verbose=0)[0][0])
    predicted_index = 1 if probability >= THRESHOLD else 0
    predicted_label = class_names[predicted_index]

    result = {
        'image_path': str(image_path),
        'predicted_label': predicted_label,
        'probability_of_positive_class': round(probability, 4),
        'threshold': THRESHOLD,
    }
    return result


if __name__ == '__main__':
    sample_path = 'data/raw/chest_xray/test/NORMAL/IM-0001-0001.jpeg'
    print(predict_image(sample_path))
