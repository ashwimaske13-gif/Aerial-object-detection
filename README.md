# 🛩️ Aerial Object Detection – Bird & Drone Detection

## 📌 Project Overview

This project implements a **deep learning-based aerial image classification and object detection system** designed to identify **birds and drones** in aerial images and determine their location within the image.

The project combines image classification and object detection techniques to build a computer vision solution for analyzing aerial imagery.

The system includes model training, prediction, evaluation, visualization of training performance, and an application interface for using the trained model.

---

## 🎯 Objectives

The main objectives of this project are:

* Identify whether an aerial image contains a bird or a drone.
* Detect the position of the object within the image.
* Apply deep learning techniques to aerial imagery.
* Train and evaluate computer vision models.
* Analyze model training and validation performance.
* Build a prediction pipeline using trained models.
* Provide an application interface for model inference.

---

## 🧠 Project Approach

The project follows a computer vision workflow:

```text
Aerial Images
      ↓
Data Preparation
      ↓
Image Preprocessing
      ↓
Model Training
      ↓
Model Validation
      ↓
Performance Analysis
      ↓
Object Classification
      ↓
Object Detection
      ↓
Prediction
```

---

## 🔍 Classification

A deep learning classification approach is used to distinguish between:

* 🐦 Bird
* 🛩️ Drone

The trained classification model learns visual patterns from aerial images and predicts the corresponding object category.

The repository contains trained TensorFlow/Keras model files that can be used for inference.

---

## 🎯 Object Detection

The project also includes an object detection component for identifying the **location of objects within aerial images**.

YOLOv8-related implementation is included in the repository for object detection.

The detector can identify objects and provide their corresponding bounding-box locations.

---

## 🤖 Deep Learning Models

The project uses deep learning approaches for image classification and object detection.

### CNN / Deep Learning

Convolutional Neural Network-based techniques are used for extracting visual features from images and performing image classification.

### YOLOv8

YOLOv8 is used for object detection to identify and localize objects in aerial images.

---

## 📊 Model Performance Analysis

The repository contains multiple training and validation visualization files for analyzing model performance, including:

* Training accuracy
* Training loss
* Validation accuracy
* Validation loss
* Accuracy gap
* Loss gap
* Epoch-wise accuracy
* Epoch-wise loss
* Accuracy comparison
* Loss comparison
* Training history

These visualizations help analyze model convergence, generalization, and potential overfitting during training.

---

## 🛠️ Technologies Used

### Programming

* Python

### Deep Learning

* TensorFlow
* Keras
* CNN
* YOLOv8

### Computer Vision

* Image Classification
* Object Detection
* Image Preprocessing

### Development

* Python scripts
* Jupyter/Notebook-based experimentation
* Application-based inference

---

## 📁 Repository Structure

```text
Aerial-object-detection/
│
├── app.py
├── train.py
├── train1.py
├── predict.py
│
├── bird_drone_model.h5
├── bird_drone_model.keras
│
├── requirement.txt
│
├── yolov8/
├── yolov8.py
│
├── 1_accuracy.png
├── 2_loss.png
├── 3_acc_gap.png
├── 4_loss_gap.png
├── 5_acc_bar.png
├── 6_loss_bar.png
├── 7_epoch_acc.png
├── 8_epoch_loss.png
├── 9_hist_acc.png
├── 10_hist_loss.png
├── 11_val_trend.png
├── 12_val_loss_trend.png
│
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ashwimaske13-gif/Aerial-object-detection.git
```

### 2. Navigate to the project

```bash
cd Aerial-object-detection
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### 4. Run the application

```bash
python app.py
```

The application can then be used to perform predictions using the trained models.

---

## 🔮 Future Enhancements

The project can be further enhanced by:

* Training on larger and more diverse aerial datasets.
* Adding more object categories.
* Improving detection accuracy for small objects.
* Applying data augmentation techniques.
* Optimizing the model for real-time inference.
* Deploying the application as a web service.
* Integrating live drone-camera input.
* Adding model performance monitoring.
* Deploying the solution on cloud infrastructure.

---

## 💼 Potential Applications

Aerial object detection can support several computer vision applications, including:

* Drone monitoring
* Wildlife monitoring
* Bird detection
* Airspace monitoring
* Aerial surveillance
* Smart transportation
* Environmental monitoring
* Security and safety applications

---

## 🎓 Project Context

This project was developed as part of an **internship project** to gain practical experience in Deep Learning, Computer Vision, Image Classification, Object Detection, and model deployment.

---

## 👩‍💻 Author

**Ashwini Maske**

Data Scientist | Machine Learning | Deep Learning | Computer Vision | Python

GitHub: https://github.com/ashwimaske13-gif
