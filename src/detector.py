import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ultralytics import YOLO


@dataclass
class DetectorConfig:
    source: str = "bus.jpg"
    weights: str = "yolo11n.pt"
    conf: float = 0.25
    device: str = "cpu"
    imgsz: int = 640
    output_dir: str = "outputs"
    run_name: str = "detection_run"


def extract_detections(result: Any) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    for box in getattr(result, "boxes", []):
        conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
        class_id = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
        label = result.names.get(class_id, str(class_id))
        detections.append({"label": label, "confidence": round(conf, 3)})
    return detections


def summarize_detections(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(detections, key=lambda d: (-d["confidence"], d["label"]))
    counts = dict(sorted(Counter(item["label"] for item in ordered).items()))
    return {"detections": ordered, "class_counts": counts, "total_detections": len(ordered)}


class YOLODetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        source_path = Path(self.config.source).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"Input source not found: {source_path}")
        self.model = YOLO(str(Path(self.config.weights).expanduser().resolve()))

    def run(self) -> Dict[str, Any]:
        output_root = Path(self.config.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        source_path = str(Path(self.config.source).expanduser().resolve())
        results = self.model(
            source=source_path,
            project=str(output_root),
            name=self.config.run_name,
            conf=self.config.conf,
            device=self.config.device,
            imgsz=self.config.imgsz,
            save=True,
            exist_ok=True,
        )

        detections: List[Dict[str, Any]] = []
        for result in results:
            detections.extend(extract_detections(result))

        summary_data = summarize_detections(detections)
        summary = {
            "source": self.config.source,
            "weights": self.config.weights,
            "confidence_threshold": self.config.conf,
            "device": self.config.device,
            "total_detections": summary_data["total_detections"],
            "detections": summary_data["detections"],
            "class_counts": summary_data["class_counts"],
            "output_directory": str(output_root / self.config.run_name),
        }

        summary_file = output_root / "summary.json"
        with summary_file.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        return summary
