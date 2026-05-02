# AI-Based Intelligent Video Surveillance Platform for Activity Recognition and Security Management in Parks

## 📌 Project Overview

This project focuses on building an **AI-based intelligent video surveillance system** for public parks to automatically detect and classify human activities using deep learning. The system is designed to identify **authorized activities** (walking/standing, running) and **unauthorized activities** (e.g., biking, vehicles) from CCTV or recorded video feeds.

The project is developed milestone-wise as part of the **Springboard Internship (Dec Batch 2025)** and follows industry-standard practices in dataset preparation, model training, validation, and deployment readiness.

---

## 🎯 Objectives

* Automate monitoring of park activities using computer vision
* Detect and classify activities in real time
* Minimize false alerts for unauthorized activities
* Build a scalable and deployable AI surveillance pipeline

---

## 🧱 Project Structure

```
park_surveillance/
│
├── data/
│   ├── raw_videos/          # Original park videos
│   ├── frames/              # Extracted video frames
│   ├── train/               # Training dataset (images & labels)
│   ├── val/                 # Validation dataset
│   └── test/                # Test dataset
│
├── src/
│   ├── preprocess.py        # Frame extraction & preprocessing
│   ├── split_dataset.py     # Train/val/test split logic
│   └── validate.py          # Dataset & class distribution checks
│
├── runs/                    # YOLO training outputs (ignored except key results)
│   └── detect/train5/       # Best training run
│       ├── results.png
│       ├── confusion_matrix.png
│       └── args.yaml
│
├── detect.py                # Inference on webcam/video
├── dataset.yaml             # YOLO dataset configuration
├── streamlit_app.py         # UI (Milestone-3)
├── .gitignore
└── README.md
```

---

## 🏁 Milestone-1: Dataset Preparation & Environment Setup

### ✔ Tasks Completed

* Project repository setup
* Python virtual environment creation
* Dependency installation (OpenCV, PyTorch, Ultralytics YOLO)
* Video data collection (park activities)
* Frame extraction using OpenCV
* Annotation using YOLO format
* Dataset preprocessing and cleaning
* Train/validation/test split

### 📂 Data Details

**Classes:**

| Class ID | Label                       |
| -------- | --------------------------- |
| 0        | Walking (includes standing) |
| 1        | Running                     |
| 2        | Unauthorized                |

---

## 🚀 Milestone-2: Model Training, Validation & Inference

### 🔹 Objective

To train and validate a robust deep learning model capable of detecting and classifying park activities with **high accuracy and low false positives**.

---

## 🧠 Model Selection

### Why YOLOv8?

* State-of-the-art real-time object detection
* High accuracy with low latency
* Supports transfer learning
* Ideal for surveillance systems

### Model Used

* **YOLOv8n (Nano)**
* Lightweight and CPU-efficient
* Suitable for real-time inference on edge devices

---

## ⚙️ Training Configuration

### Dataset Configuration (`dataset.yaml`)

* Defines dataset paths
* Specifies class labels
* Enables YOLO to load custom data

### Training Command Used

```bash
yolo detect train model=yolov8n.pt data=dataset.yaml epochs=30 imgsz=416 batch=8
```

### Parameter Explanation

| Parameter  | Description                       |
| ---------- | --------------------------------- |
| epochs=30  | Balanced training for convergence |
| imgsz=416  | Faster CPU training               |
| batch=8    | Memory-efficient                  |
| pretrained | Transfer learning enabled         |

---

## 🔄 Data Augmentation

Automatically applied by YOLOv8:

* Horizontal flip
* HSV (brightness & color variation)
* Scaling
* Mosaic augmentation

This improves robustness to:

* Lighting changes
* Camera angles
* Environmental variations

---

## 📊 Training & Validation Results

### Performance Metrics

* **mAP@0.5 ≈ 95%** (Target: >85%)
* Precision ≈ 90%+
* Recall ≈ 95%+

### Result Artifacts

* `results.png` → Training & validation curves
* `confusion_matrix.png` → Class-wise performance

These confirm:

* Strong detection accuracy
* Low false positives for unauthorized activities
* No overfitting observed

---

## 📈 Confusion Matrix Analysis

* Clear separation between walking and running
* Minimal confusion with unauthorized class
* Confirms reliability for real-world surveillance

---

## 💾 Model Saving

* Best model saved as:

```
runs/detect/train5/weights/best.pt
```

> Note: `.pt` is the correct format as YOLOv8 is PyTorch-based. TensorFlow formats (`.h5/.keras`) are not applicable.

---

## 🎥 Inference & Testing

### `detect.py`

Used to test the trained model on:

* Live webcam feed
* Pre-recorded park surveillance videos

### Capabilities

* Real-time bounding box detection
* Activity classification
* Confidence score display

---

## 🧪 Testing Summary

* Webcam inference ✔
* Video file inference ✔
* End-to-end pipeline validated ✔

---

## 📦 Git Hygiene & Best Practices

* Large artifacts (weights, cache) ignored via `.gitignore`
* Only essential files committed
* Clean, review-ready repository

---

## 🏆 Final Outcome

✔ Milestone-1 completed (data & setup)
✔ Milestone-2 completed (training, validation, inference)
✔ Target accuracy exceeded
✔ System ready for deployment phase

---

## 🔜 Next Steps (Milestone-3)

* Streamlit-based live dashboard
* Real-time alerts for unauthorized activity
* Performance optimization
* Deployment readiness

---

## 📌 Acknowledgements

* Ultralytics YOLOv8
* OpenCV
* PyTorch
* Springboard / Infosys Internship Program

---

> *This project demonstrates practical application of deep learning for intelligent surveillance systems, following industry-grade ML workflows.*
