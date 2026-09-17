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


class YOLODetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model = YOLO(str(Path(self.config.weights).expanduser().resolve()))

    def run(self) -> Dict[str, Any]:
        output_root = Path(self.config.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        run_name = "detection_run"
        results = self.model(
            source=str(Path(self.config.source).expanduser().resolve()),
            project=str(output_root),
            name=run_name,
            conf=self.config.conf,
            device=self.config.device,
            imgsz=self.config.imgsz,
            save=True,
            exist_ok=True,
        )

        detections: List[Dict[str, Any]] = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                class_id = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                label = result.names.get(class_id, str(class_id))
                detections.append(
                    {
                        "label": label,
                        "confidence": round(conf, 3),
                    }
                )

        label_counts = dict(Counter(item["label"] for item in detections))
        summary = {
            "source": self.config.source,
            "weights": self.config.weights,
            "confidence_threshold": self.config.conf,
            "device": self.config.device,
            "detections": detections,
            "class_counts": label_counts,
            "output_directory": str(output_root / run_name),
        }

        summary_file = output_root / "summary.json"
        with summary_file.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        return summary
