import numpy as np
from typing import List, Tuple, Optional
from classifier import NearestNeighborClassifier


class FeatureSelector:
    """Base class for feature selection algorithms"""
    def __init__(self):
        # Initialize tracking variables for feature selection
        self.classifier = NearestNeighborClassifier()
        self.best_features = []
        self.best_accuracy = 0.0
        self.accuracy_history = []

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

        while len(current_features) < num_features:
            best_accuracy_this_level = 0.0
            best_feature_to_add = None

            # Try each unused feature with current set
            for feature in range(num_features):
                if feature not in current_features:
                    test_features = current_features + [feature]
                    accuracy = self.classifier.evaluate(
                        features, labels, test_features, progress_callback
                    )

                    feature_str = ",".join(str(f + 1) for f in test_features)
                    print(
                        f"Using feature(s) {{{feature_str}}} accuracy is {accuracy * 100:.1f}%"
                    )

                    # Track best feature at this level
                    if accuracy > best_accuracy_this_level:
                        best_accuracy_this_level = accuracy
                        best_feature_to_add = feature

            # Add best feature found to current set
            if best_feature_to_add is not None:
                current_features.append(best_feature_to_add)
                feature_str = ",".join(str(f + 1) for f in current_features)
                print(
                    f"Feature set {{{feature_str}}} was best, "
                    f"accuracy is {best_accuracy_this_level * 100:.1f}%"
                )

                # Update overall best if improved
                if best_accuracy_this_level > self.best_accuracy:
                    self.best_accuracy = best_accuracy_this_level
                    self.best_features = current_features.copy()
                    self.accuracy_history.append(
                        (self.best_features.copy(), self.best_accuracy)
                    )

            # Stop if no improvement found
            if (
                best_feature_to_add is None
                or best_accuracy_this_level < self.best_accuracy
            ):
                break

        # Report final best feature set
        feature_str = ",".join(str(f + 1) for f in self.best_features)
        print(
            f"\nFinished search!! The best feature subset is {{{feature_str}}}, "
            f"which has an accuracy of {self.best_accuracy * 100:.1f}%"
        )

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

        # Define the baseline with all features
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

        while len(current_features) > 1:
            best_accuracy_this_level = 0.0
            worst_feature_to_remove = None

            # Try removing each feature
            for feature in current_features:
                test_features = [f for f in current_features if f != feature]
                accuracy = self.classifier.evaluate(
                    features, labels, test_features, progress_callback
                )

                self._print_feature_set(test_features, accuracy)

                # Track best accuracy when removing a feature
                if accuracy > best_accuracy_this_level:
                    best_accuracy_this_level = accuracy
                    worst_feature_to_remove = feature

            # Remove worst feature if it improves accuracy
            if worst_feature_to_remove is not None:
                current_features.remove(worst_feature_to_remove)
                self._print_feature_set(
                    current_features, best_accuracy_this_level, True
                )

                # Update overall best if improved
                if best_accuracy_this_level > self.best_accuracy:
                    self.best_accuracy = best_accuracy_this_level
                    self.best_features = current_features.copy()
                    self.accuracy_history.append(
                        (self.best_features.copy(), self.best_accuracy)
                    )
                    print("(New best!)")
                else:
                    print("(Warning: Accuracy has decreased!)")

            # Stop if no improvement found
            if (
                worst_feature_to_remove is None
                or best_accuracy_this_level < self.best_accuracy
            ):
                break

        # Report final best feature set
        feature_str = ",".join(str(f + 1) for f in self.best_features)
        print(
            f"\nFinished search!! The best feature subset is {{{feature_str}}}, "
            f"which has an accuracy of {self.best_accuracy * 100:.1f}%"
        )

        return self.best_features, self.best_accuracy, self.accuracy_history


def process_dataset(filename: str, selector: FeatureSelector):
    from data_loader import DataLoader

    loader = DataLoader()
    try:
        labels, features = loader.load_data(f"data/{filename}")
        print(
            f"This dataset has {features.shape[1]} features (not including the class attribute), "
            f"with {len(features)} instances."
        )

        def progress_update(current, total):
            pass

        return selector.select_features(
            features, labels, progress_callback=progress_update
        )

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None, None, None


def main():
    # Test both forward and backward selection
    datasets = ["CS170_Small_Data__112.txt", "CS170_Large_Data__38.txt"]

    for dataset in datasets:
        print(f"\nProcessing {dataset}")
        print("=" * 50)

        print("\nForward Selection:")
        process_dataset(dataset, ForwardSelector())

        print("\nBackward Elimination:")
        process_dataset(dataset, BackwardSelector())


if __name__ == "__main__":
    main()
