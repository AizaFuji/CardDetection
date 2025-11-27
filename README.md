# Card Detection Game

A real-time card detection system using computer vision and CNN to identify playing cards from webcam feed and simulate a simple game.

## Features

- Real-time card detection with OpenCV
- CNN-based classification for 52 card types
- Interactive game with score tracking
- Visual feedback via multiple windows

## Technologies

- Python
- OpenCV
- Keras/TensorFlow
- NumPy

## Installation

1. Clone repo: `git clone <repo-url>`
2. Install: `pip install opencv-python numpy keras tensorflow`
3. Model: `BobotKartu.h5` included; retrain with `TrainingDataKartu.py` if needed.

## Usage

Run `python Card_Detection.py`. Use spacebar to detect/open cards, 'a' for winner, 'z' to exit.

## How It Works

Preprocesses images, detects contours, classifies with CNN, and updates game scores.

