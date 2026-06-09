# Deepfakes in Visual Art --🎨 AI Art Detection using Error Level Analysis (ELA) and Convolutional Neural Networks (CNN)

## Overview

This project presents a hybrid deep learning and image forensic approach for detecting AI-generated artwork. The system combines **Error Level Analysis (ELA)** with a **Convolutional Neural Network (CNN)** to distinguish between human-created and AI-generated art images.

The project was developed as part of a mini project and is inspired by the IEEE research paper:

**"Deepfakes in Visual Art: Differentiating AI-Generated Art From Human Art Using Convolutional Neural Networks (CNN)"**

The trained model is deployed through a Flask web application that allows users to upload an image and receive a prediction along with a confidence score.

---

## Problem Statement

Recent advancements in generative AI models such as Stable Diffusion and Latent Diffusion have made AI-generated artwork increasingly realistic. As a result, distinguishing between human-created and AI-generated images has become difficult.

This project aims to provide an automated solution that can detect AI-generated artwork using forensic image analysis and deep learning techniques.

---

## Project Workflow

```text
Input Image
      │
      ▼
Error Level Analysis (ELA)
      │
      ▼
Image Preprocessing
      │
      ▼
Convolutional Neural Network (CNN)
      │
      ▼
Softmax Classification
      │
      ▼
Real Art / AI Generated Art
```

---

## Error Level Analysis (ELA)

ELA is a digital image forensic technique used to highlight compression inconsistencies within an image.

### Working

1. Original image is recompressed at 90% JPEG quality.
2. Difference between original and recompressed image is calculated.
3. Hidden artifacts become visible.
4. Resulting ELA image is passed to the CNN model.

### Why ELA?

AI-generated images often contain subtle compression artifacts and pixel-level inconsistencies that are not visible to the human eye.

ELA helps expose these hidden patterns, allowing the CNN model to learn more meaningful features.

---

## CNN Architecture

The model consists of:

* Input Layer (128 × 128 × 3)

* Conv2D (32 Filters)

* Batch Normalization

* ReLU Activation

* Max Pooling

* Conv2D (64 Filters)

* Batch Normalization

* ReLU Activation

* Max Pooling

* Conv2D (128 Filters)

* Batch Normalization

* ReLU Activation

* Max Pooling

* Conv2D (256 Filters)

* Batch Normalization

* ReLU Activation

* Max Pooling

* Flatten Layer

* Dense Layer (256 Neurons)

* Dropout Layer

* Softmax Output Layer

---

## Dataset

### AI-ArtBench Dataset

The project uses a balanced subset derived from the AI-ArtBench dataset.

Dataset Composition:

| Class            | Images |
| ---------------- | ------ |
| Real Art         | 2500   |
| AI Generated Art | 2500   |
| Total            | 5000   |

AI-generated images were created using:

* Stable Diffusion (SD)
* Latent Diffusion (LD)

Human-created artwork spans multiple artistic styles including:

* Realism
* Renaissance
* Impressionism
* Expressionism
* Surrealism
* Romanticism
* Baroque
* Art Nouveau
* Ukiyo-e
* Post-Impressionism

---

## Model Performance

### Test Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 98%   |
| Precision | 98%   |
| Recall    | 98%   |
| F1 Score  | 98%   |

### Classification Report

```text
              precision    recall  f1-score

Fake             0.98       0.97      0.98
Real             0.98       0.98      0.98

Accuracy                              0.98
```

---

## Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Data Processing

* NumPy
* Pillow (PIL)
* Scikit-Learn

### Web Application

* Flask

### Training Environment

* Google Colab
* NVIDIA T4 GPU

---

## Repository Structure

```text
.
├── app.py
├── train_ai_artbench.py
├── model_ai_artbench_run.h5
├── metrics.json
├── templates/
│   └── index.html
├── static/
│   ├── uploads/
│   └── ela_images/
├── test_dataset/
└── Colab Trained Notebook/
```

---

## Running the Project

### Clone Repository

```bash
git clone https://github.com/dedlinux777/DeepFake-Detection.git

cd DeepFake-Detection
```

### Install Dependencies

```bash
pip install tensorflow flask pillow numpy scikit-learn
```

### Run Flask Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Upload an image and view the prediction result.

---

## Future Enhancements

* Train on complete AI-ArtBench dataset
* Integrate advanced architectures such as ResNet and EfficientNet
* Extend to video deepfake detection
* Add explainable AI visualizations
* Deploy as a cloud-hosted web service

---

## Reference

Tinago, N., Verkijika, S. F., & Mamabolo, K. E.

**Deepfakes in Visual Art: Differentiating AI-Generated Art From Human Art Using Convolutional Neural Networks (CNN)**

IEEE Access, 2025.

DOI: 10.1109/ACCESS.2025.3596882

---
