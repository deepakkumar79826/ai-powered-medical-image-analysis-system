from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
PLOTS_DIR = OUTPUTS_DIR / 'plots'
PREDICTIONS_DIR = OUTPUTS_DIR / 'predictions'

# Dataset folders expected from the Chest X-Ray Pneumonia dataset
TRAIN_DIR = RAW_DATA_DIR / 'chest_xray' / 'train'
VAL_DIR = RAW_DATA_DIR / 'chest_xray' / 'val'
TEST_DIR = RAW_DATA_DIR / 'chest_xray' / 'test'

# Training configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 1e-4
SEED = 42

# Output files
BEST_MODEL_PATH = MODELS_DIR / 'best_model.keras'
FINAL_MODEL_PATH = MODELS_DIR / 'final_model.keras'
CLASS_NAMES_PATH = MODELS_DIR / 'class_names.txt'
HISTORY_JSON_PATH = OUTPUTS_DIR / 'training_history.json'
METRICS_JSON_PATH = OUTPUTS_DIR / 'metrics.json'
CONFUSION_MATRIX_PATH = PLOTS_DIR / 'confusion_matrix.png'
TRAINING_CURVES_PATH = PLOTS_DIR / 'training_curves.png'
SAMPLE_PREDICTIONS_PATH = PLOTS_DIR / 'sample_predictions.png'

# Inference threshold
THRESHOLD = 0.5
