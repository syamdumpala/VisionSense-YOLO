# VisionSense YOLO

<p align="center">
  <img src="https://github.com/syamdumpala/VisionSense-YOLO/actions/workflows/ci.yml/badge.svg" alt="CI status" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/YOLO-Ultralytics-00D3FF?logo=yolo&logoColor=white" alt="YOLO Ultralytics" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status Active" />
</p>

VisionSense YOLO is an end-to-end computer vision project built with Ultralytics YOLO for object detection in images and multi-image inputs. It detects common objects, saves annotated outputs, and generates structured detection summaries that can be used in ML workflows, dashboards, or portfolio projects.

## Overview

This project demonstrates a production-style object detection pipeline:

- load a YOLO model
- run inference on an image or folder of images
- filter results using a confidence threshold
- save detection outputs and metadata
- export a JSON summary for downstream analysis
- expose a simple CLI for easy experimentation

It is designed for both learning and portfolio use, with a GitHub-ready structure and a clear workflow for future enhancements.

## Features

- YOLOv11 nano object detection
- image and folder-based inference support
- adjustable confidence threshold
- output directory management
- JSON summary export with class counts
- CPU/GPU-friendly device configuration
- clean Python package structure for scaling up

## Tech stack

- Python 3.10+
- Ultralytics YOLO
- OpenCV
- NumPy
- GitHub Actions for CI

## Project structure

```text
VisionSense-YOLO/
├── .github/
│   └── workflows/
│       └── ci.yml
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── bus.jpg
├── yolo11n.pt
├── src/
│   ├── __init__.py
│   └── detector.py
├── tests/
│   └── test_detector.py
└── outputs/
    └── detection_run/
```

## Quick start

### 1) Clone the repository

```bash
git clone https://github.com/syamdumpala/VisionSense-YOLO.git
cd VisionSense-YOLO
```

### 2) Set up the environment

Using the provided local environment:

```powershell
cd d:\yolo_tutorial
.\YOLO_tutorial\Scripts\python.exe -m pip install -r requirements.txt
```

Or create your own virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Run the detector

```powershell
cd d:\yolo_tutorial
.\YOLO_tutorial\Scripts\python.exe main.py --source bus.jpg --output-dir outputs --conf 0.25
```

### 4) Run on a folder of images

```powershell
.\YOLO_tutorial\Scripts\python.exe main.py --source data/images --output-dir outputs --conf 0.3
```

## Example output

The script generates:

- a labeled detection image in the run folder
- a summary JSON file with each detected label and confidence
- a terminal log showing the detected objects and total counts

Example console output:

```text
Detection completed. Results saved to: outputs\detection_run
- bus (0.94)
- person (0.888)
- person (0.878)
Total detections: 5
Summary JSON: outputs\summary.json
```

## Configuration options

```powershell
.\YOLO_tutorial\Scripts\python.exe main.py --source bus.jpg --weights yolo11n.pt --conf 0.35 --device cpu --imgsz 640 --output-dir outputs
```

Available options:

- `--source`: image, folder, or video path
- `--weights`: custom YOLO weights file
- `--conf`: detection threshold
- `--device`: `cpu` or `cuda`
- `--imgsz`: inference image size
- `--output-dir`: output directory name
- `--run-name`: output subfolder name

## CI and quality checks

This project includes a GitHub Actions workflow to run automated checks on every push and pull request. The badge at the top of this README reflects the status of that workflow.

## Roadmap

- add video-based detection support with frame export
- support batch inference from a dataset folder
- integrate a small Flask API or web dashboard
- add benchmarking and evaluation metrics
- improve packaging and deployment for real-world usage

## License

This project is intended for learning, experimentation, and portfolio use. Add an appropriate open-source license if you plan to publish it publicly.

## Repository

- GitHub: https://github.com/syamdumpala/VisionSense-YOLO

## Author

Syam D. P. | Computer Vision & AI project
