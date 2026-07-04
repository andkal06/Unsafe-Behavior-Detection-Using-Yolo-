# Unsafe Behavior Detection Using YOLO

A real-time unsafe behavior detection system built with **YOLO (Ultralytics)**, capable of detecting **smoking**, **vaping**, and **phone use** from images, video files, webcam feeds, or RTSP streams (CCTV).

When any of these behaviors is detected, the system automatically triggers an **audio alert** as a warning.

## Features

- **Single script for all modes**: image, video, webcam, and RTSP stream — the mode is auto-detected from `--source` (can also be set manually).
- **Automatic audio alert** when a target class is detected (default: `smoking`, `vaping`, `Phone`), with a cooldown to prevent continuous alerting on every frame.
- **Custom alert sound** support — use your own audio file (`.mp3`/`.wav`) or fall back to a built-in system beep.
- **Debug mode** to print every class detected in each frame.
- **Save detection results** to a file (`.jpg` for images, `.mp4` for video/stream).
- Displays **real-time FPS** in video/stream mode.

## Repository Structure

```
.
├── best.pt              # Trained YOLO model weights
├── detect_realtime.py   # Main detection script (image/video/webcam/RTSP)
└── .gitignore
```

## Model and Dataset

The model was trained using **YOLO (Ultralytics)** on a combined dataset from [Roboflow Universe](https://universe.roboflow.com/):

| Dataset | Classes | Source |
|---|---|---|
| Smoking & Vape Detection | `smoking`, `vaping` | [livestreamdetection / smoking-vape-detection-7nvky](https://universe.roboflow.com/livestreamdetection/smoking-vape-detection-7nvky/dataset/1) |
| Phone Use Detection | `Phone` | [nakarin-samon / phone-use](https://universe.roboflow.com/nakarin-samon/phone-use/dataset/7) |

Both datasets were merged and used to train a single YOLO object detection model capable of recognizing all three behaviors (`best.pt`).

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/andkal06/Unsafe-Behavior-Detection-Using-Yolo-.git
   cd Unsafe-Behavior-Detection-Using-Yolo-
   ```

2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install ultralytics opencv-python
   ```

   If you want to use a custom alert sound file (`--alert-sound`), also install:
   ```bash
   pip install pygame
   ```

## Usage

Run `detect_realtime.py` with the mode that fits your use case:

**Detection on an image:**
```bash
python detect_realtime.py --weights best.pt --source "images (1).jpg"
```

**Detection on a video file:**
```bash
python detect_realtime.py --weights best.pt --source video.mp4
```

**Detection via webcam:**
```bash
python detect_realtime.py --weights best.pt --source 0
```

**Detection via RTSP stream (CCTV):**
```bash
python detect_realtime.py --weights best.pt --source rtsp://user:pass@ip:554/stream --save
```

**Custom alert classes and cooldown:**
```bash
python detect_realtime.py --weights best.pt --source 0 --alert-classes smoking,Phone --alert-cooldown 5
```

**Disable audio alert:**
```bash
python detect_realtime.py --weights best.pt --source 0 --no-alert
```

Press **`q`** in the video window to exit stream mode, or press any key to close the window in image mode.
