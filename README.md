# Sign Language Detection using Deep Learning

## Overview
This project implements a Hand Gesture Recognition System using computer vision and deep learning. The system captures hand gestures in real-time, processes them with Mediapipe, and classifies them into predefined categories using a trained PyTorch model.

## Features
- **Real-Time Hand Gesture Detection**: Uses Mediapipe to extract hand landmarks from live video.
- **Customizable Dataset**: Collects and processes gesture data for training.
- **Deep Learning Model**: Implements a PyTorch-based neural network for gesture classification.
- **Interactive Inference**: Displays bounding boxes and predicted gestures live on the webcam feed.

## Requirements
- Python >= 3.7  
- Libraries:
  - opencv-python  
  - mediapipe  
  - numpy  
  - torch  
  - scikit-learn  
  - pickle  

## Project Structure
- `data_collection.py`: Captures gesture data from webcam and stores it in labeled folders.
- `data_processing.py`: Extracts and normalizes hand landmarks using Mediapipe, stores them in a `.pickle` file.
- `model_training.py`: Trains a PyTorch neural network on the processed data.
- `model_testing.py`: Evaluates model performance on a test set.
- `live_inference.py`: Performs real-time gesture recognition from webcam input.
- `data/`: Directory where raw gesture images are stored.
- `data.pickle`: Processed and structured hand landmark data.
- `model.pth`: Trained PyTorch model.

## Usage Instructions

### Data Collection
Start the data collection script:

```
python data_collection.py
```

Use the webcam to capture gestures for each class. Press `Q` when ready to start capturing data.  
Each class’s data is stored in a separate folder within `./data`.

### Data Processing
Run the data processing script:

```
python data_processing.py
```

This extracts hand landmarks using Mediapipe, normalizes them, and saves the data in `data.pickle`.

### Model Training
Train the gesture classification model:

```
python model_training.py
```

The trained model is saved as `model.pth`.

### Real-Time Inference
Run the live inference script:

```
python live_inference.py
```

The script displays the webcam feed with predictions and bounding boxes for recognized gestures. Press `Q` to quit.

## Customization

- **Number of Classes**: Adjust the `number_of_classes` variable in the data collection script.
- **Dataset Size**: Modify `dataset_size` to control the number of samples per class.
- **Model Parameters**: Change the input size, hidden size, and output size in the model training script to suit your dataset.

## Key Components

- **Mediapipe Hands**: Detects hand landmarks and connections.
- **PyTorch Model**: A simple feedforward neural network for gesture classification.
- **Label Mapping**: Customize labels for gestures in the `labels_dict` dictionary.

## Example Output

Real-time webcam feed with:
- **Bounding Box**: Highlights detected hands.
- **Predicted Gesture**: Displays the recognized gesture above the bounding box.

## Challenges and Improvements

- **Lighting and Backgrounds**: Train the model with diverse lighting conditions and backgrounds for robustness.
- **Class Overlap**: Ensure gestures are visually distinct for better accuracy.
- **Additional Classes**: Expand the dataset to include more gestures.

## Acknowledgments

This project leverages:
- **Mediapipe** for efficient hand tracking and landmark detection.
- **PyTorch** for building and training the neural network.

