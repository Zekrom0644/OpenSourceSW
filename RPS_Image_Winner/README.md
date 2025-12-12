# Rock-Paper-Scissors Image Winner 🎮✋

## 📌 Project Overview
This project detects two hands in a single image and determines
the Rock-Paper-Scissors result using OpenCV-based image processing.

The system analyzes hand contours, estimates finger gaps,
classifies each hand as ROCK, PAPER, or SCISSORS,
and finally judges the winner.

## 🛠 Technologies Used
- Python 3.10
- OpenCV
- NumPy

## 📂 Project Structure
RPS_Image_Winner/
├── images/
│ └── test.jpg
├── results/
│ └── rps_result.jpg
├── rps_winner_cv.py
└── README.md

perl
코드 복사

## ▶️ How to Run
```bash
pip install opencv-python numpy
python rps_winner_cv.py
🧠 Algorithm Description
Skin-color segmentation in HSV space

Contour detection and selection of two largest hand regions

Convex hull & convexity defects analysis

Finger gap counting to classify hand gesture

Rule-based winner decision

📸 Result Example
The output image displays:

Detected hand regions

Classified gestures (ROCK / PAPER / SCISSORS)

Final game result (LEFT WINS / RIGHT WINS / DRAW)

✨ Notes
Works best with simple backgrounds and clear hand poses