# Unsafe Behavior Detection Using YOLO

A YOLO-based computer vision system for detecting unsafe workplace behaviors from images, video files, webcams, or CCTV/RTSP streams. The model detects three classes: **smoking**, **vaping**, and **Phone** (phone use).

## Overview

This project was developed as part of a computer vision internship focused on workplace safety monitoring. The model is trained to identify potentially unsafe activities in real time, with an optional audible alert system for live monitoring use cases.

## Model Details

- **Architecture:** YOLOv8s
- **Input size:** 640x640
- **Classes:** `smoking`, `vaping`, `Phone`
- **Training epochs:** 50
- **Training data:** Combined and rebalanced dataset from multiple Roboflow sources (smoking/vaping detection dataset and phone-use detection dataset), with additional augmentation applied to the underrepresented `vaping` class to reduce class imbalance.

### Validation Metrics

| Metric | Value |
|---|---|
| Precision | 0.736 |
| Recall | 0.626 |
| mAP50 | 0.620 |
| mAP50-95 | 0.330 |

### Known Limitations

- Performance is strongest for the `Phone` class and weakest for `vaping`, reflecting the relative amount of training data available per class.
- The model is trained primarily on overhead/CCTV-style imagery of indoor workspaces. It may not generalize well to significantly different camera angles or contexts (for example, close-up dashboard-camera footage), since these fall outside the training data distribution.
- The model detects specific objects/actions only. It does not perform person detection or face detection; a bounding box will only appear when one of the three trained classes is visible in the frame.

## Repository Structure

```
.
├── detect.py       # Unified detection script (image, video, webcam, RTSP)
├── best.pt         # Trained model weights
└── README.md
```

## Installation

Requires Python 3.9 or later.

```bash
pip install ultralytics opencv-python
```

Optional, for the alert sound features:

```bash
pip install pygame pydub imageio-ffmpeg
```

- `pygame` is used to play a custom alert sound file during live detection.
- `pydub` and `imageio-ffmpeg` are used to embed alert sounds into saved output videos. `imageio-ffmpeg` provides a bundled ffmpeg binary automatically, so no separate ffmpeg installation or PATH configuration is required.

## Usage

The script automatically detects whether the source is an image, a video file, a webcam, or a stream, based on the `--source` argument.

### Single image

```bash
python detect.py --weights best.pt --source "image.jpg"
```

### Video file

```bash
python detect.py --weights best.pt --source video.mp4
```

### Webcam

```bash
python detect.py --weights best.pt --source 0
```

### RTSP / CCTV stream

```bash
python detect.py --weights best.pt --source rtsp://user:pass@ip:554/stream
```

### Saving output

Adding `--output <name>` automatically saves the result:

```bash
python detect.py --weights best.pt --source video.mp4 --output result
```

- For video/stream sources, this produces `result.mp4` and, if the alert dependencies are installed, `result_with_audio.mp4` containing an embedded alert sound at each detection timestamp.
- For image sources, this produces `result.jpg`.

## Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--weights` | `best.pt` | Path to the trained model weights file |
| `--source` | `0` | `0`/`1`... for webcam index, or a path/URL for image, video, or RTSP stream |
| `--mode` | `auto` | Force `image` or `stream` mode if auto-detection is incorrect |
| `--conf` | `0.4` | Confidence threshold |
| `--imgsz` | `640` | Inference image size |
| `--device` | (auto) | `cpu` or GPU index (e.g. `0`) |
| `--save` | off | Save output (implied automatically if `--output` is set) |
| `--output` | none | Output file name (without extension); also enables saving |
| `--no-alert` | off | Disable the alert system entirely |
| `--alert-classes` | `smoking,vaping,Phone` | Comma-separated list of classes that trigger an alert |
| `--alert-cooldown` | `3.0` | Minimum seconds between consecutive alerts |
| `--alert-sound` | none | Path to a custom `.mp3`/`.wav` alert sound. If not set, a synthetic tone is generated automatically. |
| `--debug` | off | Print every detected class and confidence per frame, useful for diagnosing detection issues |

## Alert System

During video, webcam, or stream processing, the script plays an audible alert whenever one of the classes listed in `--alert-classes` is detected, subject to the cooldown period. If no custom sound file is provided, a short synthetic tone is generated automatically, so no additional setup is required for basic use.

When saving output with `--output`, the same alerts are also embedded into a separate `_with_audio` video file, timed to match the moment each detection occurred in the source footage.

## Training Notes

The training dataset was built by combining two separate Roboflow projects, remapping and merging their label sets into the final three classes. To address a substantial class imbalance between `Phone` and the combined `smoking`/`vaping` data, class-aware sampling was applied when constructing the final training set, together with additional targeted augmentation for the `vaping` class specifically.
