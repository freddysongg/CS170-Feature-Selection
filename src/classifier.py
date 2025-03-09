import numpy as np
from typing import List, Optional
import sys


class NearestNeighborClassifier:
    """Base class for nearest neighbor classifier"""
    def __init__(self):
        # Store the training data
        self.train_features = None
        self.train_labels = None
        self.feature_subset = None

    def euclidean_distance(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        feature_indices: Optional[List[int]] = None,
    ) -> float:
        # Use only the selected features if specified
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
        # Use either the provided or stored training data
        features = train_features if train_features is not None else self.train_features
        labels = train_labels if train_labels is not None else self.train_labels

        # Find the nearest neighbor
        distances = [
            self.euclidean_distance(test_instance, features[i], feature_indices)
            for i in range(len(features))
        ]
        nearest_index = np.argmin(distances)
        return labels[nearest_index]
        
    # Perform leave-one-out cross validation
    def cross_validation(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_indices: Optional[List[int]] = None,
        progress_callback: Optional[callable] = None,
    ) -> float:
        correct_predictions = 0
        n_instances = len(features)

        for i in range(n_instances):
            if progress_callback:
                progress_callback(i + 1, n_instances)

            # Remove the current instance for training
            train_features = np.delete(features, i, axis=0)
            train_labels = np.delete(labels, i)

            # Classify the current instance and check if it's correct
            predicted_label = self.classify(
                features[i], train_features, train_labels, feature_indices
            )

            if predicted_label == labels[i]:
                correct_predictions += 1

        return correct_predictions / n_instances

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_indices: List[int],
        progress_callback: Optional[callable] = None,
    ) -> float:
        return self.cross_validation(
            features, labels, feature_indices, progress_callback
        )


def main():
    from data_loader import DataLoader

    loader = DataLoader()
    classifier = NearestNeighborClassifier()

    datasets = ["CS170_Small_Data__112.txt", "CS170_Large_Data__38.txt"]

    for dataset in datasets:
        try:
            print(f"\nEvaluating classifier on {dataset}")
            print("=" * 50)

            labels, features = loader.load_data(f"data/{dataset}")
            print(f"Features: {features.shape[1]}, Instances: {len(features)}")

            def progress_update(current, total):
                sys.stdout.write(f"\rEvaluating instance {current}/{total}")
                sys.stdout.flush()

            accuracy = classifier.cross_validation(
                features, labels, progress_callback=progress_update
            )

            print(f"\nClassification accuracy: {accuracy:.4f}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")


if __name__ == "__main__":
    main()
