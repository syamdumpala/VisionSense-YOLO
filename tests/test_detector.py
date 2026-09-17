import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detector import DetectorConfig, YOLODetector


def test_detector_runs_on_sample_image():
    source = Path("bus.jpg")
    assert source.exists(), "Sample image is missing"

    config = DetectorConfig(
        source="bus.jpg",
        weights="yolo11n.pt",
        conf=0.25,
        device="cpu",
        imgsz=640,
        output_dir="outputs/test_run",
        run_name="pytest_run",
    )

    detector = YOLODetector(config)
    summary = detector.run()

    assert summary["total_detections"] > 0
    assert "detections" in summary
    assert summary["output_directory"].endswith("pytest_run")
