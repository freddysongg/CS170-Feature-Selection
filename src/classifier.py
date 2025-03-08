import numpy as np
from typing import Tuple, List, Optional
import sys


class NearestNeighborClassifier:
    def __init__(self):
        self.train_features = None
        self.train_labels = None
        self.feature_subset = None

    def euclidean_distance(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        feature_indices: Optional[List[int]] = None,
    ) -> float:
        if feature_indices is not None:
            x1 = x1[feature_indices]
            x2 = x2[feature_indices]
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def classify(
        self,
        test_instance: np.ndarray,
        train_features: Optional[np.ndarray] = None,
        train_labels: Optional[np.ndarray] = None,
        feature_indices: Optional[List[int]] = None,
    ) -> int:
        features = train_features if train_features is not None else self.train_features
        labels = train_labels if train_labels is not None else self.train_labels

        distances = [
            self.euclidean_distance(test_instance, features[i], feature_indices)
            for i in range(len(features))
        ]
        nearest_index = np.argmin(distances)
        return labels[nearest_index]

    def cross_validation(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_indices: Optional[List[int]] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[float, List[int]]:
        correct_predictions = 0
        misclassified_indices = []
        n_instances = len(features)

        for i in range(n_instances):
            if progress_callback:
                progress_callback(i + 1, n_instances)

            train_features = np.delete(features, i, axis=0)
            train_labels = np.delete(labels, i)

            predicted_label = self.classify(
                features[i], train_features, train_labels, feature_indices
            )

            if predicted_label == labels[i]:
                correct_predictions += 1
            else:
                misclassified_indices.append(i)

        accuracy = correct_predictions / n_instances
        return accuracy, misclassified_indices

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray, 
        feature_indices: List[int],
        progress_callback: Optional[callable] = None,
    ) -> float:
        accuracy, _ = self.cross_validation(
            features, labels, feature_indices, progress_callback
        )
        return accuracy


def main():
    from data_loader import DataLoader

    loader = DataLoader()
    folder_path = "data/"

    try:
        datasets = loader.load_datasets(folder_path)
        classifier = NearestNeighborClassifier()

        for size in ["small", "large"]:
            print(f"\n{size.upper()} DATASETS EVALUATION")
            print("=" * 50)

            for idx, (labels, features) in enumerate(datasets[size]["data"]):
                dataset_name = datasets[size]["files"][idx]
                print(f"\nEvaluating dataset: {dataset_name}")
                print(f"Features: {features.shape[1]}, Instances: {len(features)}")

                def progress_update(current, total):
                    sys.stdout.write(
                        f"\rProgress: {current}/{total} instances evaluated"
                    )
                    sys.stdout.flush()

                accuracy, misclassified = classifier.cross_validation(
                    features, labels, progress_callback=progress_update
                )

                print(f"\nResults for {dataset_name}:")
                print(f"Classification accuracy: {accuracy:.4f}")
                print(f"Number of misclassifications: {len(misclassified)}")
                print("-" * 50)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
