Gesture Controlled Snake Game (Python + Computer Vision)

A real-time Snake Game controlled entirely using hand gestures captured from a webcam.

Instead of a keyboard, the snake follows the direction of the user's index finger movement detected using computer vision. The system processes live video frames, extracts hand landmark coordinates, interprets direction, and converts it into game actions.

This project demonstrates how raw visual data can be captured, processed, and transformed into meaningful real-time decisions.

---

Demo

The snake moves based on finger direction (left, right, up, down) using a webcam feed.

---

Tech Stack

- Python
- OpenCV
- MediaPipe (Hand Tracking)
- NumPy
- Pygame

---

How It Works

1. Webcam captures live video frames.
2. MediaPipe detects 21 hand landmarks.
3. Index finger tip coordinates are extracted.
4. Movement direction is calculated using coordinate changes.
5. Direction is mapped to game controls.
6. Snake moves in real time.

This forms a simple real-time data pipeline:

Camera Input → Feature Extraction → Direction Prediction → Game Action

---

Skills Demonstrated

- Real-time data processing
- Feature extraction from video streams
- Coordinate and spatial data handling
- Event-driven programming
- Human-computer interaction
- Computer vision integration in Python

---

Installation

1. Clone the repository

git clone https://github.com/sarrthak-1/python-snake-game.git
cd python-snake-game

2. Install dependencies

pip install opencv-python mediapipe pygame numpy

3. Run the project

python snake_gesture.py

---

Controls

Move your index finger in front of the webcam:

- Move Right → Snake goes Right
- Move Left → Snake goes Left
- Move Up → Snake goes Up
- Move Down → Snake goes Down

No keyboard required.

---

Future Improvements

- Gesture based pause/resume
- Multi-hand support
- Score tracking dashboard
- Gesture classification model
- Data logging for analytics

---

Why This Project Matters (For Data Analytics)

Data analytics is not only about analyzing existing datasets — it also involves creating and capturing data.

In this project:

- Each video frame acts as raw data
- Hand landmarks are extracted features
- Direction detection is a real-time decision system

This shows the complete cycle:
Data Collection → Processing → Interpretation → Action

---

Author

Sarthak

I am an aspiring Data Analyst exploring Python, real-time data processing, and computer vision applications.