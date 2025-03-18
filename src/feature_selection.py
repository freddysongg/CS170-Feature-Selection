import numpy as np
from typing import List, Tuple, Optional
from classifier import NearestNeighborClassifier
from data_loader import DataLoader
from metrics_collector import MetricsCollector, SelectionMetrics
import time


class FeatureSelector:
    """Base class for feature selection algorithms"""

    def __init__(self):
        # Initialize tracking variables for feature selection
        self.classifier = NearestNeighborClassifier()
        self.best_features = []
        self.best_accuracy = 0.0
        self.accuracy_history = []
        self.metrics_collector = MetricsCollector()
        self.individual_feature_accuracies = {}

    def _print_feature_set(
        self, features: List[int], accuracy: float, is_best: bool = False
    ):
        feature_str = ",".join(str(f + 1) for f in features)
        if is_best:
            print(
                f"Feature set {{{feature_str}}} was best, "
                f"accuracy is {accuracy * 100:.1f}%"
            )
        else:
            print(
                f"Using feature(s) {{{feature_str}}} accuracy is {accuracy * 100:.1f}%"
            )

    def _record_metrics(self, dataset_name: str, features: np.ndarray, runtime: float):
        accuracy_history = [
            {"feature_count": len(features), "accuracy": acc}
            for features, acc in self.accuracy_history
        ]

        # Convert weak features to list of tuples
        weak_features = []
        for feature, accuracy in self.individual_feature_accuracies.items():
            if (
                accuracy > 0.7 * self.best_accuracy
                and feature not in self.best_features
            ):
                weak_features.append((feature, accuracy))

        metrics = SelectionMetrics(
            dataset_name=dataset_name,
            algorithm_name=self.__class__.__name__,
            dataset_size=len(features),
            total_features=features.shape[1],
            runtime=runtime,
            accuracy_history=accuracy_history,
            best_accuracy=self.best_accuracy,
            best_feature_count=len(self.best_features),
            best_features=self.best_features,
            individual_feature_accuracies=self.individual_feature_accuracies,
            weak_features=weak_features,
        )

        self.metrics_collector.save_metrics(metrics)

    def _analyze_weak_features(self, features: np.ndarray, labels: np.ndarray):
        """Identify features that perform well individually but get overshadowed in combinations"""
        print("\nIndividual feature performance:")

        # Get individual feature accuracies
        for feature in range(features.shape[1]):
            accuracy = self.classifier.evaluate(features, labels, [feature])
            self.individual_feature_accuracies[feature] = accuracy
            print(f"Feature {feature + 1} individual accuracy: {accuracy * 100:.1f}%")

        # Identify weak features
        weak_features = []
        for feature, accuracy in self.individual_feature_accuracies.items():
            if (
                accuracy > 0.7 * self.best_accuracy
                and feature not in self.best_features
            ):
                weak_features.append((feature, accuracy))

        if weak_features:
            print("\nWeak features:")
            for feature, accuracy in sorted(
                weak_features, key=lambda x: x[1], reverse=True
            ):
                print(
                    f"Feature {feature + 1}: {accuracy * 100:.1f}% accuracy individually"
                )


class ForwardSelector(FeatureSelector):
    """Forward selection algorithm"""

    def select_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[List[int], float, List[Tuple[List[int], float]]]:
        num_features = features.shape[1]
        current_features = []
        self.accuracy_history = []

        # Define the baseline with all features
        all_features = list(range(num_features))
        initial_accuracy = self.classifier.evaluate(
            features, labels, all_features, progress_callback
        )
        print(
            f"\nRunning nearest neighbor with all {num_features} features, using "
            f"'leaving-one-out' evaluation, I get an accuracy of {initial_accuracy * 100:.1f}%"
        )

        print("Beginning search.")

        # Start with empty feature set as baseline
        accuracy = self.classifier.evaluate(features, labels, [], progress_callback)
        self.best_accuracy = accuracy
        self.best_features = []
        self.accuracy_history.append(([], accuracy))

        # Continue until we've tried all features
        while len(current_features) < num_features:
            best_accuracy_this_level = 0.0
            best_feature_to_add = None

            for feature in range(num_features):
                if feature not in current_features:
                    test_features = current_features + [feature]
                    accuracy = self.classifier.evaluate(
                        features, labels, test_features, progress_callback
                    )

                    print(
                        f"Using feature(s) {{{','.join(str(f + 1) for f in test_features)}}} "
                        f"accuracy is {accuracy * 100:.1f}%"
                    )

                    if accuracy > best_accuracy_this_level:
                        best_accuracy_this_level = accuracy
                        best_feature_to_add = feature

            if best_feature_to_add is not None:
                current_features.append(best_feature_to_add)
                print(
                    f"Feature set {{{','.join(str(f + 1) for f in current_features)}}} "
                    f"was best, accuracy is {best_accuracy_this_level * 100:.1f}%"
                )

                self.accuracy_history.append(
                    (current_features.copy(), best_accuracy_this_level)
                )

                # Update overall best if improved
                if best_accuracy_this_level > self.best_accuracy:
                    self.best_accuracy = best_accuracy_this_level
                    self.best_features = current_features.copy()
                    print("(New best!)")
                else:
                    print("(Warning: Accuracy has decreased!)")

        feature_str = ",".join(str(f + 1) for f in self.best_features)
        print(
            f"\nFinished search!! The best feature subset is {{{feature_str}}}, "
            f"which has an accuracy of {self.best_accuracy * 100:.1f}%"
        )

        # Analyze weak features
        self._analyze_weak_features(features, labels)
        return self.best_features, self.best_accuracy, self.accuracy_history


class BackwardSelector(FeatureSelector):
    """Backward elimination algorithm"""

    def select_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[List[int], float, List[Tuple[List[int], float]]]:
        num_features = features.shape[1]
        current_features = list(range(num_features))
        self.accuracy_history = []

        initial_accuracy = self.classifier.evaluate(
            features, labels, current_features, progress_callback
        )
        print(
            f"\nRunning nearest neighbor with all {num_features} features, using "
            f"'leaving-one-out' evaluation, I get an accuracy of {initial_accuracy * 100:.1f}%"
        )

        print("Beginning search.")

        # Start with all features as baseline
        self.best_accuracy = initial_accuracy
        self.best_features = current_features.copy()
        self.accuracy_history.append((self.best_features.copy(), self.best_accuracy))

        # Continue until only one feature remains
        while len(current_features) > 1:
            best_accuracy_this_level = 0.0
            worst_feature_to_remove = None

            for feature in current_features:
                test_features = [f for f in current_features if f != feature]
                accuracy = self.classifier.evaluate(
                    features, labels, test_features, progress_callback
                )

                self._print_feature_set(test_features, accuracy)

                if accuracy > best_accuracy_this_level:
                    best_accuracy_this_level = accuracy
                    worst_feature_to_remove = feature

            if worst_feature_to_remove is not None:
                current_features.remove(worst_feature_to_remove)
                self._print_feature_set(
                    current_features, best_accuracy_this_level, True
                )

                self.accuracy_history.append(
                    (current_features.copy(), best_accuracy_this_level)
                )

                if best_accuracy_this_level > self.best_accuracy:
                    self.best_accuracy = best_accuracy_this_level
                    self.best_features = current_features.copy()
                    print("(New best!)")
                else:
                    print("(Warning: Accuracy has decreased!)")

        feature_str = ",".join(str(f + 1) for f in self.best_features)
        print(
            f"\nFinished search!! The best feature subset is {{{feature_str}}}, "
            f"which has an accuracy of {self.best_accuracy * 100:.1f}%"
        )

        self._analyze_weak_features(features, labels)
        return self.best_features, self.best_accuracy, self.accuracy_history


def process_dataset(filename: str, selector: FeatureSelector):
    loader = DataLoader()
    try:
        labels, features = loader.load_data(f"data/{filename}")
        print(
            f"This dataset has {features.shape[1]} features (not including the class attribute), "
            f"with {len(features)} instances."
        )

        start_time = time.time()
        result = selector.select_features(features, labels)
        runtime = time.time() - start_time

        # Record metrics
        selector._record_metrics(filename, features, runtime)

        return result

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None, None, None


def main():
    datasets = ["CS170_Large_Data__12.txt", "CS170_Large_Data__17.txt"]

    print("\nWelcome to Feature Selection Algorithm")
    print("1. Forward Selection")
    print("2. Backward Elimination")

    while True:
        try:
            choice = int(input("\nPlease select your algorithm (1 or 2): "))
            if choice in [1, 2]:
                break
            print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number (1 or 2).")

    selector = ForwardSelector() if choice == 1 else BackwardSelector()
    algorithm_name = "Forward Selection" if choice == 1 else "Backward Elimination"

    for dataset in datasets:
        print(f"\nProcessing {dataset} using {algorithm_name}")
        print("=" * 50)
        process_dataset(dataset, selector)


if __name__ == "__main__":
    main()
