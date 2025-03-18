from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import os


@dataclass
class SelectionMetrics:
    dataset_name: str
    algorithm_name: str
    dataset_size: int
    total_features: int
    runtime: float
    accuracy_history: List[Dict[str, float]]
    best_accuracy: float
    best_feature_count: int
    best_features: List[int]
    individual_feature_accuracies: Dict[int, float]
    weak_features: List[Tuple[int, float]]


class MetricsCollector:
    def __init__(self, output_dir="metrics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_metrics(self, metrics: SelectionMetrics):
        output_file = os.path.join(
            self.output_dir,
            f"{metrics.dataset_name}_{metrics.algorithm_name}_metrics.json",
        )

        data = {
            "dataset_name": metrics.dataset_name,
            "algorithm": metrics.algorithm_name,
            "dataset_size": metrics.dataset_size,
            "total_features": metrics.total_features,
            "runtime": metrics.runtime,
            "accuracy_history": metrics.accuracy_history,
            "best_accuracy": metrics.best_accuracy,
            "best_feature_count": metrics.best_feature_count,
            "best_features": metrics.best_features,
            "individual_feature_accuracies": metrics.individual_feature_accuracies,
            "weak_features": metrics.weak_features,
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
