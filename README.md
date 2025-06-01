# X-ray Baggage Anomaly Detection using YOLOv8

This project trains a YOLOv8 object detection model to identify anomalies (e.g., weapons, sharp objects) in X-ray baggage images. The goal is to build a fast and accurate model that can assist in airport security or similar critical areas.

## 🎯 Objective

To detect dangerous and prohibited items such as guns and knives hidden inside luggage by training a custom object detection model on X-ray images.

## 🧠 Approach

- 🔍 YOLOv8 from [Ultralytics](https://github.com/ultralytics/ultralytics)
- 🗂️ Custom `data.yaml` for dataset configuration
- 📊 Training with visualized metrics (Precision, Recall, mAP)
- 🔁 Transfer learning using pretrained weights (e.g., `yolov8s.pt`)
- 🛑 Early stopping enabled

## 📂 Dataset

This project uses the publicly available X-ray baggage dataset from Kaggle:

🔗 [X-ray Baggage Anomaly Detection Dataset](https://www.kaggle.com/datasets/orvile/x-ray-baggage-anomaly-detection/code)

The dataset includes annotated images and is suitable for object detection tasks. A corresponding `data.yaml` file is provided for training.

## 🛠️ Requirements

- Python >= 3.8
- Ultralytics YOLOv8 (`pip install ultralytics`)
- PyTorch (GPU recommended)
- OpenCV, Matplotlib, Pandas, PIL

## 🚀 Training

To start training:

```bash
python yolov8_xray_training.py


