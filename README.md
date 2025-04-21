# Hand Gesture Recognition System

---

## Project Overview

This repository contains a Hand Gesture Recognition System built with computer vision and deep learning techniques. It captures hand movements in real time, processes them using Mediapipe, and classifies gestures into predefined categories with a PyTorch-based model.

---

## Key Features

- **Live Gesture Detection**: Detects and processes hand gestures from webcam input using Mediapipe.
- **Custom Dataset Creation**: Allows users to collect and label gesture data for model training.
- **Neural Network Model**: Utilizes a PyTorch feedforward neural network for accurate gesture classification.
- **Real-Time Visualization**: Displays detected hands with bounding boxes and predicted gesture labels on the video feed.

---

## System Requirements

- Python 3.7 or higher  
- Required libraries:  
  - `opencv-python`  
  - `mediapipe`  
  - `numpy`  
  - `torch`  
  - `scikit-learn`  
  - `pickle`  

Install dependencies with:  
```bash
pip install opencv-python mediapipe numpy torch scikit-learn
