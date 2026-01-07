# 🫁 ChestX-ray14 Multi-Label Classification  
### ECE 460J Final Project

Multi-label chest X-ray disease classification using deep learning and transfer learning, with a focus on class imbalance, threshold tuning, and interpretability.

---

## 📌 Overview

Chest X-rays are one of the most common medical imaging exams, but interpreting them accurately requires experience and careful attention to subtle visual cues. In real clinical settings, radiologists must process large volumes of scans, which can lead to heavy workload, missed findings, and delayed diagnoses.

This project explores whether deep learning models can assist by predicting thoracic diseases from a single frontal chest X-ray. Using the NIH ChestX-ray14 dataset, the project examines how modern neural networks behave under real-world challenges such as noisy labels, extreme class imbalance, and overlapping disease patterns.

---

## 🎯 Task Description

Given a chest X-ray image, the model outputs a probability for each disease label. Because multiple diseases can appear in a single scan, this is a **multi-label classification** problem (sigmoid outputs rather than softmax).

**Diseases considered include:**

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

The primary implemented pipeline in this repository focuses on **EfficientNet-B0**, with other architectures explored through supporting experiments.

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
- Labels automatically extracted from radiology reports  
- Strong class imbalance  
- Multi-label structure  

### Preprocessing steps
- Convert disease strings → multi-hot label vectors  
- Resize images to 224×224  
- Convert images to PyTorch tensors  
- Train/validation split (80/20)  

---

## 🔥 Training Considerations

### Handling Class Imbalance
Mitigation strategies:

- `pos_weight` in `BCEWithLogitsLoss`  
- Focal loss in selected trials  
- Avoiding accuracy as a metric  

### Threshold Tuning

A default threshold of 0.5 is too strict for rare conditions.  
Per-disease thresholds were tuned between **0.05–0.40** to maximize F1 score.

### Interpretability (Grad-CAM)

Observed patterns:

- True positives → strong lung-field activation  
- False positives → ribs/heart borders  
- Overlapping diseases → mixed attention regions  

---

## 🧪 Practical Training Choices

### Using subsets during development
To keep training manageable without a GPU, subsets (5k–10k images) were used for rapid experimentation.

### Epoch selection
- EfficientNet-B0: **8 epochs**  
- Others tested up to **12–14 epochs**

### Compute constraints
All training done on CPU-only or limited GPU access.  
Batch sizes + subset sizes adjusted accordingly.

---

## 📊 Results Snapshot

General behavior of multi-label CXR models:

- AUC appears strong even when F1 is low  
- Rare diseases remain challenging  
- EfficientNet and DenseNet are stable  
- Lightweight models perform well given size  

### Expected performance ranges (full dataset)

- **Macro AUC:** 0.78–0.81  
- **Macro F1:** 0.20–0.30  

### My reproduced results (subset, CPU only)

- **Macro AUC:** ≈ 0.73  
- **Macro F1:** ≈ 0.19  

These match expected values under limited compute.

---

## ▶️ Running the Project

> **Important:** The dataset is NOT included.  
> Download via KaggleHub or NIH first.

### Folder structure (required)

/path/to/chestxray14/
Data_Entry_2017.csv
images_001/
images/
00000001_000.png
00000002_001.png
images_002/
images/
images_003/
images/

yaml
Copy code

---

### 1️⃣ Create your environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
2️⃣ Train the model
bash
Copy code
PYTHONPATH=. python scripts/train.py \
  --csv_path "/path/to/chestxray14/Data_Entry_2017.csv" \
  --img_root "/path/to/chestxray14" \
  --epochs 8 \
  --batch_size 32 \
  --lr 1e-4 \
  --train_subset 10000 \
  --val_subset 2000
This produces:

Copy code
model_best.pth
model_last.pth
3️⃣ Evaluate the model
bash
Copy code
PYTHONPATH=. python scripts/eval.py \
  --csv_path "/path/to/chestxray14/Data_Entry_2017.csv" \
  --img_root "/path/to/chestxray14" \
  --weights model_best.pth \
  --subset 2000
Metrics saved to:

bash
Copy code
results/tables/metrics.txt
4️⃣ Generate Grad-CAM visualizations
bash
Copy code
PYTHONPATH=. python scripts/gradcam.py \
  --csv_path "/path/to/chestxray14/Data_Entry_2017.csv" \
  --img_root "/path/to/chestxray14" \
  --weights model_best.pth \
  --samples 8
Images saved to:

bash
Copy code
results/gradcam/
