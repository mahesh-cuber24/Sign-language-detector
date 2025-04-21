Hand Gesture Recognition System

Project Overview
This repository contains a Hand Gesture Recognition System built with computer vision and deep learning techniques. It captures hand movements in real time, processes them using Mediapipe, and classifies gestures into predefined categories with a PyTorch-based model.

Key Features

Live Gesture Detection: Detects and processes hand gestures from webcam input using Mediapipe.
Custom Dataset Creation: Allows users to collect and label gesture data for model training.
Neural Network Model: Utilizes a PyTorch feedforward neural network for accurate gesture classification.
Real-Time Visualization: Displays detected hands with bounding boxes and predicted gesture labels on the video feed.


System Requirements

Python 3.7 or higher
Required libraries:
opencv-python
mediapipe
numpy
torch
scikit-learn
pickle



Install dependencies with:
pip install opencv-python mediapipe numpy torch scikit-learn


Repository Structure

Data Collection: Gathers gesture images and labels them for training.
Data Preprocessing: Normalizes hand landmarks extracted via Mediapipe and saves them in a structured format.
Model Training: Trains a PyTorch neural network using the processed dataset.
Model Evaluation: Tests the model’s accuracy on a separate test dataset.
Live Inference: Performs real-time gesture recognition on webcam input.


How to Use

Collect Gesture Data

Run the data collection script:
python collect_data.py


Use the webcam to record gestures for each class. Press 'Q' to begin capturing.

Data is saved in separate folders under ./data.



Process Data

Execute the preprocessing script:
python preprocess_data.py


This generates normalized hand landmarks and stores them in data.pickle.



Train the Model

Start the training script:
python train_model.py


The trained model is saved as model.pth.



Run Live Inference

Launch the inference script:
python run_inference.py


View real-time gesture predictions with bounding boxes. Press 'Q' to exit.





Customization Options

Gesture Classes: Modify number_of_classes in the data collection script to define gesture categories.
Dataset Size: Adjust dataset_size to increase or decrease samples per class.
Model Architecture: Tune input, hidden, and output layer sizes in the training script to match your dataset.


Core Components

Mediapipe Hands: Provides robust hand landmark detection and tracking.
PyTorch Neural Network: A feedforward model for classifying gestures.
Label Customization: Define gesture labels in the labels_dict dictionary.


Example Output
The live webcam feed shows:

Bounding Box: Outlines detected hands.
Gesture Label: Displays the predicted gesture above the hand.


Potential Improvements

Robustness: Train with varied lighting and backgrounds to improve performance.
Distinct Gestures: Ensure gestures are unique to avoid classification errors.
Expanded Gestures: Add more gesture classes to the dataset.


Acknowledgments
This project builds on:

Mediapipe for hand tracking and landmark extraction.
PyTorch for neural network development and training.


License
Licensed under the MIT License. Contributions are encouraged!

Contact
For inquiries or contributions, please create an issue or submit a pull request on GitHub.
