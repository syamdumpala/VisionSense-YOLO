import argparse
from pathlib import Path

from src.detector import DetectorConfig, YOLODetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on an image and save labeled output."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="bus.jpg",
        help="Path to the input image or video file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="Model weights file to use for detection.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for filtering detections.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where detection outputs are saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computing device to use: cpu or cuda.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size used by the model during inference.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = DetectorConfig(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        device=args.device,
        imgsz=args.imgsz,
        output_dir=args.output_dir,
    )

    detector = YOLODetector(config)
    summary = detector.run()

    print(f"\nDetection completed. Results saved to: {summary['output_directory']}")
    if summary["detections"]:
        for item in summary["detections"]:
            print(f"- {item['label']} ({item['confidence']})")
    else:
        print("- No objects were detected above the confidence threshold.")

    summary_path = Path(config.output_dir) / "summary.json"
    print(f"Summary JSON: {summary_path}")
