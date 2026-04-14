# AI-Powered Medical Image Analysis System

A beginner-friendly yet industry-oriented computer vision project that analyzes chest X-ray images and predicts whether the image is **NORMAL** or **PNEUMONIA**.

> **Important:** This project is for **education, portfolio, and research demonstration only**. It is **not a medical device** and must not be used for real clinical diagnosis.

---

## 1. Project Overview
Medical imaging is used in hospitals, diagnostic labs, and radiology centers to support disease detection. In this project, we simulate a realistic healthcare AI workflow using a public chest X-ray dataset and a transfer learning model based on **MobileNetV2**.

### What this project does
- Loads public chest X-ray data
- Preprocesses medical images
- Trains a deep learning classifier
- Evaluates the model using accuracy, balanced accuracy, confusion matrix, and classification report
- Predicts disease class for a new image
- Saves visual proof for GitHub and interviews

---

## 2. Problem Statement
Manual image review can be time-consuming, and human review may vary from one expert to another. This project shows how AI can assist radiology workflows by quickly classifying chest X-ray images into:
- **NORMAL**
- **PNEUMONIA**

The goal is not to replace doctors. The goal is to build a system that acts like an **AI assistant** for image triage and educational proof-of-work.

---

## 3. Industry Relevance
This kind of pipeline is relevant to:
- Hospitals
- Diagnostic labs
- Radiology centers
- Health-tech startups
- AI healthcare product teams

It demonstrates the same core building blocks used in real medical imaging products:
- image ingestion
- preprocessing
- model inference
- evaluation
- visual reporting
- reproducible ML workflow

---

## 4. Selected Approach
### Why this version?
This repository uses **transfer learning with MobileNetV2** because it gives a strong balance of:
- easier execution
- better performance than a tiny CNN baseline
- professional GitHub value
- practical understanding of computer vision pipelines

### Task
**Binary classification** on chest X-ray images:
- Class 0: `NORMAL`
- Class 1: `PNEUMONIA`

---

## 5. Tech Stack
- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **scikit-learn**
- **Pillow**

---

## 6. Dataset
Recommended dataset:
- **Chest X-Ray Images (Pneumonia)**
- Download from Kaggle and extract into:

```text
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Do **not** upload the raw dataset to GitHub because it is large. Keep it local and only upload your code, screenshots, plots, and documentation.

---

## 7. Project Architecture
```text
Medical Images
    ↓
Data Loading
    ↓
Preprocessing (resize, RGB conversion, batching)
    ↓
Augmentation (flip, rotation, zoom)
    ↓
Transfer Learning Model (MobileNetV2)
    ↓
Classifier Head (Dense + Sigmoid)
    ↓
Prediction
    ↓
Evaluation + Visualization
    ↓
Saved Outputs for GitHub
```

---

## 8. Folder Structure
```text
AI-Medical-Image-Analysis/
│
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── outputs/
│   ├── plots/
│   └── predictions/
├── images/
├── docs/
├── README.md
├── requirements.txt
├── .gitignore
└── main.py
```

---

## 9. Installation
### Create virtual environment
#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 10. How to Run
### Train the model
```bash
python main.py --mode train
```

### Evaluate the model
```bash
python main.py --mode evaluate
```

### Predict on one image
```bash
python main.py --mode predict --image data/raw/chest_xray/test/NORMAL/IM-0001-0001.jpeg
```

---

## 11. Expected Outputs
After running training and evaluation, you should get:
- `models/best_model.keras`
- `models/final_model.keras`
- `models/class_names.txt`
- `outputs/training_history.json`
- `outputs/final_test_metrics_after_train.json`
- `outputs/metrics.json`
- `outputs/plots/training_curves.png`
- `outputs/plots/confusion_matrix.png`
- `outputs/plots/sample_predictions.png`

These files become your portfolio proof.

---

## 12. Suggested Screenshots for GitHub
Capture and upload these to the `images/` folder:
- dataset folder structure
- sample X-ray images
- terminal showing training start
- training accuracy/loss plot
- confusion matrix plot
- sample prediction output
- repository homepage screenshot

---

## 13. GitHub Upload Strategy
### Create repository
Suggested repo name:
```text
AI-Powered-Medical-Image-Analysis-System
```

### Useful description
```text
AI-powered chest X-ray image analysis system using TensorFlow, OpenCV, and transfer learning for pneumonia detection.
```

### Example Git commands
```bash
git init
git add .
git commit -m "Initial project structure"
git branch -M main
git remote add origin https://github.com/your-username/AI-Powered-Medical-Image-Analysis-System.git
git push -u origin main
```

---

## 14. Commit Plan
- `setup: create project structure and requirements`
- `data: add dataset instructions and loading pipeline`
- `model: add MobileNetV2 transfer learning classifier`
- `train: add training pipeline with callbacks`
- `eval: add confusion matrix and classification report`
- `predict: add single-image inference script`
- `docs: improve README and screenshots`

---

## 15. Learning Outcomes
By building this project, you will understand:
- medical image classification workflow
- image preprocessing for deep learning
- transfer learning using MobileNetV2
- training and evaluation in TensorFlow
- confusion matrix and balanced accuracy
- how to package an AI project for GitHub and interviews

---

## 16. Future Improvements
You can extend this project by adding:
- Grad-CAM heatmaps
- Streamlit web app
- Flask API
- multi-class disease classification
- object detection on the RSNA dataset
- model explainability report
- Docker support

---

## 17. Interview Pitch
> “I built an AI-powered medical image analysis system using public chest X-ray data. The project uses TensorFlow, OpenCV, and transfer learning with MobileNetV2 to classify pneumonia vs normal images. I designed the full ML pipeline including preprocessing, training, evaluation, confusion matrix visualization, and single-image inference. I also structured it as a clean GitHub project with outputs and documentation for portfolio proof.”

---

## 18. Disclaimer
This repository is for **educational and research demonstration only**. It does not provide medical advice, and it should not be used in real patient care.
