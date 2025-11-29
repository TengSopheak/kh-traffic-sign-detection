# Cambodian Traffic Sign Detection System 🚦

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8m-green)
![Roboflow](https://img.shields.io/badge/Data-Roboflow-purple)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

## 📖 Overview

This project implements a computer vision system designed to detect and classify traffic signs specific to the Cambodian road environment. Utilizing the **YOLOv8m (Medium)** architecture, the model is trained to recognize 10 distinct classes of traffic signs. This system serves as a proof-of-concept for Advanced Driver Assistance Systems (ADAS) or autonomous navigation modules tailored for local infrastructure.

## 📂 Dataset

The dataset consists of locally sourced imagery of Cambodian traffic signs. To address class imbalance and improve model generalization in varying lighting/orientations, extensive data augmentation was applied via **Roboflow**.

### Class Distribution
The dataset contains the following 10 classes:

| Class Name | Image Count (Original) |
| :--- | :--- |
| **Speed Limit 40** | 63 |
| **Curve Left** | 60 |
| **Speed Limit 30** | 54 |
| **Curve Right** | 50 |
| **Double Curve (First Right)** | 43 |
| **Double Curve (First Left)** | 39 |
| **No Entry** | 24 |
| **Speed Limit 60** | 23 |
| **T-Junction Ahead** | 20 |
| **Speed Limit 80** | 15 |

### Preprocessing & Augmentation Strategy
To ensure robustness against real-world driving conditions, the following transformations were applied to the training set:

* **Crop:** 0% Minimum Zoom, 15% Maximum Zoom
* **Rotation:** ±15°
* **Shear:** ±15° Horizontal, ±15° Vertical
* **Color Space Adjustments:**
    * **Hue:** ±10°
    * **Saturation:** ±15%
    * **Brightness:** -20% to +0%
    * **Exposure:** ±10%
* **Image Quality:**
    * **Blur:** Up to 1px (Simulating motion blur)
    * **Noise:** Up to 1.01% of pixels (Simulating camera grain/low light)

## 🧠 Model Architecture

This project utilizes **YOLOv8m (You Only Look Once - Version 8 Medium)**.

* **Base Weights:** Pre-trained on COCO dataset (Transfer Learning).
* **Architecture:** PyTorch backend.
* **Reasoning:** The 'Medium' variant was selected to strike an optimal balance between inference speed (FPS) and mean Average Precision (mAP), suitable for the complex visual clutter often found in street environments.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/cambodia-traffic-sign-detection.git](https://github.com/yourusername/cambodia-traffic-sign-detection.git)
    cd cambodia-traffic-sign-detection
    ```

2.  **Install Dependencies**
    Ensure you have Python 3.8+ installed.
    ```bash
    pip install ultralytics roboflow
    # Add other requirements if necessary (pandas, numpy, matplotlib, notebook)
    pip install -r requirements.txt
    ```

## 🚀 Usage

### Training
To retrain the model with the dataset configuration:
```bash
yolo task=detect mode=train model=yolov8m.pt data=data.yaml epochs=100 imgsz=640
```

## 📊 Results and Evaluation

The YOLOv8m model was trained for **50 epochs** on the augmented Cambodian traffic sign dataset. The model demonstrates strong performance across the validation set, validating the choice of architecture and data augmentation strategy.

### Final Validation Metrics
The following metrics represent the final performance of the model on the validation dataset (Metrics derived from Epoch 35, where the highest mAP@50-95 was observed):

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Precision (P)** | **0.891** | 89.1% of all bounding box detections were correct. |
| **Recall (R)** | **0.827** | 82.7% of all ground-truth signs were successfully detected. |
| **mAP@50 (IoU 0.50)** | **0.901** | High overall performance in localizing and classifying signs with a standard Intersection over Union threshold. |
| **mAP@50-95 (Average)** | **0.839** | Indicates robust performance across stringent IoU thresholds (0.50 to 0.95), crucial for production-level accuracy. |

**Performance Analysis:**
The high mAP@50 and mAP@50-95 values confirm the model's strong capability for accurate traffic sign detection. The minor difference between Precision and Recall suggests that while the model is highly confident in its predictions (high Precision), it still occasionally misses some signs (lower Recall), likely due to the inherent data imbalance observed in the original dataset for minority classes (e.g., 'Speed Limit 80', 'T-Junction Ahead').

---

## 🔮 Future Work

To evolve this project into a production-grade system and mitigate current limitations, the following steps are recommended for continued development:

1.  **Strategic Data Acquisition & Balancing:** Prioritize collecting and integrating additional, diverse image data specifically for underrepresented classes, such as **Speed Limit 80** and **T-Junction Ahead**. This will directly improve the model's recall and robustness, minimizing detection failures for these signs.
2.  **Model Optimization for Edge Deployment:** Investigate techniques for model quantization (e.g., conversion to **ONNX** or **TensorRT**) to significantly reduce model size and inference latency. This is necessary for deployment on resource-constrained edge devices (e.g., **NVIDIA Jetson**, mobile platforms) required for real-time ADAS applications.
3.  **Implementation of Object Tracking:** Integrate a temporal tracking algorithm (e.g., DeepSORT or BoT-SORT) to maintain sign identity across consecutive video frames. This post-processing step stabilizes detections, reduces visual flickering, and improves the reliability of the system in dynamic driving environments.
4.  **Environmental Robustness Testing:** Conduct focused validation and improvement efforts on performance under challenging conditions (e.g., low light, heavy rain, glare), which often challenge the generalization capacity of vision models.
