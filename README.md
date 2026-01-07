🫁 ChestX-ray14 Multi-Label Classification  
ECE 460J Final Project

Multi-label chest X-ray disease classification using deep learning and transfer learning, with a focus on class imbalance, threshold tuning, and interpretability.

---

## 📌 Overview

Chest X-rays are one of the most common medical imaging exams, but interpreting them accurately requires experience and careful attention to subtle visual cues. In real clinical settings, radiologists must process large volumes of scans, which can lead to heavy workload, missed findings, and delayed diagnoses.

This project explores whether deep learning models can assist by predicting thoracic diseases from a single frontal chest X-ray. Using the NIH ChestX-ray14 dataset, the project examines how modern neural networks behave under real-world challenges such as noisy labels, extreme class imbalance, and overlapping disease patterns.

---

## 🎯 Task Description

Given a chest X-ray image, the model outputs a probability for each disease label. Because multiple diseases can appear in a single scan, this is a **multi-label classification** problem (sigmoid outputs rather than softmax).

Diseases considered include:

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis,  
Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax, Hernia

---

## 🧠 Models Explored

Several architectures were explored throughout the project to understand tradeoffs between performance, stability, and complexity under imbalance:

- EfficientNet-B0  
- DenseNet-121  
- ResNet-18  
- VGG16  
- MobileNetV2  
- Vision Transformer (ViT-B/16)

The primary implemented pipeline in this repository focuses on **EfficientNet-B0**, with other architectures explored through supporting experiments to compare model behavior.

---

## 🧰 Tech Stack

- Python  
- PyTorch / Torchvision  
- NumPy, Pandas, scikit-learn  
- Grad-CAM for visual explanations  
- KaggleHub for dataset access  

---

## 🗂 Dataset & Preprocessing

**Dataset:** NIH ChestX-ray14  
112,120 frontal chest X-rays with up to 14 disease labels per image.

### Key challenges
- Labels are automatically extracted from radiology reports (not hand-labeled)
- Severe class imbalance across disease categories
- Multi-label structure with overlapping conditions

### Preprocessing steps
- Convert disease strings into multi-hot label vectors  
- Resize images to 224×224  
- Convert images to PyTorch tensors  
- Random train/validation split (80/20)

---

## 🔥 Training Considerations

### Handling Class Imbalance
A common failure mode is predicting “no disease” for every image.

Mitigation strategies included:
- `pos_weight` in `BCEWithLogitsLoss` to upweight rare diseases
- Focal loss in selected experiments
- Avoiding accuracy as a primary metric due to imbalance

### Threshold Tuning
A default threshold of 0.5 is often too strict for rare diseases.

Per-disease thresholds were tuned over a range (typically 0.05–0.40) to maximize validation F1 score. This step consistently improved recall for rare conditions.

### Interpretability (Grad-CAM)
Grad-CAM was used to visualize regions contributing to model predictions.

Observed patterns:
- Correct predictions often focus on lung fields
- Some false positives attend to ribs, heart borders, or diaphragm
- Multi-label cases may emphasize one abnormality while missing another

### 🧪 Practical Training Choices

Training on a large medical imaging dataset involves tradeoffs between runtime, experimentation speed, and model behavior. To keep the project practical while still meaningful, a few choices were made during training.

**Using subsets during development**  
Although ChestX-ray14 contains over 112,000 images, many experiments were run on smaller random subsets (typically 5k–10k images). This made it much faster to iterate on data loading, loss functions, and threshold tuning. Subsets were sampled from the full dataset so the original class imbalance was preserved.

**Epoch selection**  
Different models converged at different speeds:

- **EfficientNet-B0** was trained for **8 epochs**, since validation performance stabilized early.
- Other architectures (e.g., DenseNet-121, ResNet variants) were tested for **12–14 epochs** in supporting experiments to see whether longer training improved rare disease recall.

The goal was to stop training once validation behavior stabilized rather than overfitting to training loss.

**Compute constraints**  
All experiments were run under realistic hardware limits (CPU-only or limited GPU access). Training parameters such as subset size, batch size, and number of epochs were chosen to reflect what is feasible in an academic setting while still allowing fair model comparisons.


---

📊 Results Snapshot (High-Level)

Because of dataset noise and imbalance, emphasis was placed on macro-level metrics and per-class behavior rather than raw accuracy.

General trends:

- AUC can appear strong even when decision thresholds perform poorly  
- F1 score is the most challenging metric for rare diseases  
- EfficientNet and DenseNet showed stable performance across classes  
- Lightweight models remained competitive given their size  

Typical tuned performance ranges:

- Macro AUC: ~0.78–0.81  
- Macro F1: ~0.20–0.30  

⭐ Additional Notes on My Implementation:
- Due to CPU-only training and reduced subset size, my reproduced EfficientNet-B0 run reached **Macro AUC ≈ 0.73** and **Macro F1 ≈ 0.19**, which is consistent with expected behavior under limited compute. Published full-dataset baselines for EfficientNet-B0 typically report **Macro AUC around 0.78–0.81**.
- 
▶️ Running the Project

Note:
The NIH ChestX-ray14 dataset is not included in this repository.
You’ll need to download it separately (for example through KaggleHub or the official NIH website).

Once you have the dataset, make sure your folder looks like this:

/path/to/chestxray14/
    Data_Entry_2017.csv
    images_001/
        images/
            00000001_000.png
            00000002_001.png
            ...
    images_002/
        images/
    images_003/
        images/
    ...

1. Create your environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Train the model

Replace the paths with your dataset location:

PYTHONPATH=. python scripts/train.py \
  --csv_path "/path/to/chestxray14/Data_Entry_2017.csv" \
  --img_root "/path/to/chestxray14" \
  --epochs 8 \
  --batch_size 32 \
  --lr 1e-4 \
  --train_subset 10000 \
  --val_subset 2000


This will train EfficientNet-B0 and produce two saved weights:

model_best.pth
model_last.pth

3. Evaluate the model
PYTHONPATH=. python scripts/eval.py \
  --csv_path "/path/to/chestxray14/Data_Entry_2017.csv" \
  --img_root "/path/to/chestxray14" \
  --weights model_best.pth \
  --subset 2000


This will output macro-level scores and save a results file here:

results/tables/metrics.txt

4. Generate Grad-CAM visualizations
PYTHONPATH=. python scripts/gradcam.py \
  --csv_path "/path/to/chestxray14/Data_Entry_2017.csv" \
  --img_root "/path/to/chestxray14" \
  --weights model_best.pth \
  --samples 8


Images will appear in:

results/gradcam/
