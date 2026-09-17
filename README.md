# YOLO Object Detection Project

A production-style computer vision project built around Ultralytics YOLO for object detection using a sample street-scene image. The project is structured so it can be pushed to GitHub later and expanded into a more complete ML pipeline.

## Overview

This project demonstrates an end-to-end object detection workflow:

- loading a YOLO model
- running inference on an image
- generating detection output
- saving visualization and metadata
- exposing a simple command-line interface for re-use

The sample model and image included in this workspace are ready to run immediately, making this a good starting point for a portfolio-ready ML project.

## Why this project

This repository is designed to be:

- easy to run locally
- easy to extend for more images or videos
- cleanly structured for GitHub hosting
- suitable as a base for a future portfolio or deployment project

## Project structure

```text
YOLO-object-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── src/
│   ├── __init__.py
│   └── detector.py
├── bus.jpg
├── yolo11n.pt
└── outputs/
    └── detection_run/
```

## Tech stack

- Python 3.10+
- Ultralytics YOLO
- OpenCV
- NumPy

## Quick start

### 1) Create or activate your Python environment

If you are using the local environment already available in this workspace:

```powershell
cd d:\yolo_tutorial
.\YOLO_tutorial\Scripts\python.exe -m pip install -r requirements.txt
```

If you prefer your own virtual environment:

```powershell
cd d:\yolo_tutorial
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2) Run the detector

```powershell
cd d:\yolo_tutorial
.\YOLO_tutorial\Scripts\python.exe main.py --source bus.jpg --output-dir outputs
```

Or with a custom threshold:

```powershell
.\YOLO_tutorial\Scripts\python.exe main.py --source bus.jpg --conf 0.35 --output-dir outputs
```

## Example output

The script will generate:

- a detected image in the output folder
- a `summary.json` file with the detected labels and confidence values
- a clean CLI summary in the terminal

Example console output:

```text
Detection completed. Results saved to: outputs\detection_run
- person (0.89)
- bus (0.96)
Summary JSON: outputs\summary.json
```

## Customization

You can easily change:

- input image or video path
- detection confidence threshold
- output directory
- model weights file
- target device (`cpu` or `cuda`)

Examples:

```powershell
.\YOLO_tutorial\Scripts\python.exe main.py --source my_image.jpg --weights my_model.pt --conf 0.4 --device cpu
```

## GitHub repository

When you are ready to push this project to GitHub, add your repository URL here:

```text
https://github.com/your-username/your-repository.git
```

## Roadmap

- add support for video input
- add batch inference over a folder
- create a web dashboard or Flask API
- add model benchmarking and metrics
- add CI/CD and automated checks

## License

This project is intended for learning and experimentation. Add your preferred license later if you plan to publish it publicly.
