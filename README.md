# SolarWatch: AI-Powered Rooftop Solar Audit

**Team:** Aura Farming
**Model:** YOLO11-Medium + SAHI
**Accuracy:** ~97% F1 Score

## 📖 Overview
SolarWatch is an automated pipeline designed to audit rooftop solar installations for the PM Surya Ghar scheme. Unlike basic detection models that overestimate area, our system uses **Statistical Bounding Box Correction** to mathematically estimate the true footprint of rotated arrays, ensuring high-accuracy quantification ($m^2$) without the computational overhead of segmentation.

## 🛠️ Tech Stack
* **Core Model:** YOLO11-Medium (Standard Detection)
* **Enhancement:** SAHI (Slicing Aided Hyper Inference) + EDSR Super-Resolution
* **Logic:** Geometric Filtering & Statistical Area Correction (Fill Factor 0.85)
* **Deployment:** Dockerized Python Container

## 🚀 How to Run (Docker)
1.  **Build the Image:**
    ```bash
    docker build -f "Pipeline code/Dockerfile" -t solar-watch .
    ```

2.  **Run Inference:**
    ```bash
    docker run -v $(pwd):/app solar-watch
    ```

## 📊 Performance
* **F1 Score:** 97.7%
* **Recall:** 99.6% (Minimizes missed subsidies)
* **Quantification:** Applies a statistical correction factor (0.85) to axis-aligned boxes, solving the diagonal panel overestimation problem.

## 📂 Folder Structure
* `Pipeline code`: Inference scripts and Docker setup.
* `Trained model file`: The fine-tuned YOLO11 weights.
* `Model Training Logs`: Evidence of training convergence (Loss/mAP curves).