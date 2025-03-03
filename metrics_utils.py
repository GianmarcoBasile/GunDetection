import json
import os
from pathlib import Path


def save_training_metrics(metrics, model_name):
    """Save training metrics to JSON file"""
    metrics_dir = Path("metrics/training")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    output_file = metrics_dir / f"{model_name}.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)


def save_testing_metrics(metrics, model_name):
    """Save testing metrics to JSON file"""
    metrics_dir = Path("metrics/testing")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    output_file = metrics_dir / f"{model_name}.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(model_name, phase="training"):
    """Load metrics from JSON file"""
    metrics_dir = Path(f"metrics/{phase}")
    input_file = metrics_dir / f"{model_name}.json"

    if not input_file.exists():
        return None

    with open(input_file, "r") as f:
        return json.load(f)


class MetricsLogger:
    def __init__(self, model_name):
        self.model_name = model_name
        self.training_metrics = {
            "epochs": [],
            "train_loss": [],
            "learning_rate": [],
            "val_map": [],
            "val_map_50": [],
            "val_map_75": [],
        }

    def log_epoch(self, epoch_num, train_loss, lr, coco_evaluator=None):
        """Log metrics for one epoch"""
        self.training_metrics["epochs"].append(epoch_num)
        self.training_metrics["train_loss"].append(float(train_loss))
        self.training_metrics["learning_rate"].append(float(lr))

        if coco_evaluator is not None:
            stats = coco_evaluator.coco_eval["bbox"].stats
            self.training_metrics["val_map"].append(
                float(stats[0])
            )  # mAP @ IoU=0.50:0.95
            self.training_metrics["val_map_50"].append(
                float(stats[1])
            )  # mAP @ IoU=0.50
            self.training_metrics["val_map_75"].append(
                float(stats[2])
            )  # mAP @ IoU=0.75

    def save_training_metrics(self):
        """Save accumulated training metrics"""
        save_training_metrics(self.training_metrics, self.model_name)

    def log_testing_metrics(self, coco_evaluator, inference_times):
        """Log and save testing metrics"""
        stats = coco_evaluator.coco_eval["bbox"].stats
        testing_metrics = {
            "map": float(stats[0]),  # mAP @ IoU=0.50:0.95
            "map_50": float(stats[1]),  # mAP @ IoU=0.50
            "map_75": float(stats[2]),  # mAP @ IoU=0.75
            "inference_times": inference_times,
        }
        save_testing_metrics(testing_metrics, self.model_name)
