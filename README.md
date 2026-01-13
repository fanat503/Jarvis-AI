# Jarvis AI Assistant (v0.1) 

This is a computer vision project powered by **OpenCV** and **MediaPipe**.
The assistant can track faces, apply filters (nose mask) and take selfies using hand gestures.

## Features 🌟
- **Face Tracking**: Real-time face detection with a clown nose filter.
- **Hand Gestures**: Show a "pinch" gesture (thumb + index finger) to take a selfie.
- **Multithreading**: Smooth video capture without lag.

## Requirements 
- Python 3.x
- OpenCV
- MediaPipe
- NumPy

## How to run 
1. Install libraries:
   ```bash
   pip install opencv-python mediapipe numpy


Controls 
'n' - Toggle Nose filter
'b' - Toggle Face Blur
'q' - Quit   