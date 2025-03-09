import numpy as np
from pathlib import Path
import sys
import os
import re


class DataLoader:
    """Loads and validates data from a file"""
    def __init__(self):
        # Track loaded dataset information
        self.labels = None
        self.features = None
        self.num_features = 0
        self.num_instances = 0
        self.current_dataset = None

    def load_data(self, filename):
        try:
            if not Path(filename).exists():
                raise FileNotFoundError(f"Data file not found: {filename}")

            data = np.loadtxt(filename)

            if data.shape[1] < 2:
                raise ValueError(
                    "Data must have at least one feature and a class label"
                )

            # Split into features and labels
            self.labels = data[:, 0]
            self.features = data[:, 1:]

            # Validate data format
            if not np.all(np.isin(self.labels, [1, 2])):
                raise ValueError("Class labels must be either 1 or 2")

            self.num_instances, self.num_features = self.features.shape

            if np.any(np.isnan(self.features)):
                raise ValueError("Dataset contains missing values")

            self.current_dataset = os.path.basename(filename)

            return self.labels, self.features

        except Exception as e:
            print(f"Error loading data: {str(e)}", file=sys.stderr)
            raise

    def get_stats(self):
        if self.features is None or self.labels is None:
            raise ValueError("No data has been loaded yet")

        stats = {
            "dataset_name": self.current_dataset,
            "num_instances": self.num_instances,
            "num_features": self.num_features,
            "class_distribution": np.bincount(self.labels.astype(int))[1:],
            "feature_means": np.mean(self.features, axis=0),
            "feature_stds": np.std(self.features, axis=0),
        }
        return stats

    def load_datasets(self, folder_path, dataset_type=None):
        datasets = {
            "small": {"files": [], "data": []},
            "large": {"files": [], "data": []},
        }

        pattern = r"CS170_(Small|Large)_Data__\d+\.txt"

        try:
            for filename in sorted(os.listdir(folder_path)):
                match = re.match(pattern, filename)
                if match:
                    size = match.group(1).lower()

                    if dataset_type and size != dataset_type.lower():
                        continue

                    file_path = os.path.join(folder_path, filename)
                    labels, features = self.load_data(file_path)

                    datasets[size]["files"].append(filename)
                    datasets[size]["data"].append((labels, features))

            if not any(datasets[k]["files"] for k in datasets):
                raise FileNotFoundError(
                    f"No valid data files found in folder: {folder_path}"
                )

            return datasets

        except Exception as e:
            print(f"Error loading datasets: {str(e)}", file=sys.stderr)
            raise

    def get_dataset_summary(self, datasets):
        summary = {"small": [], "large": []}

        for size in ["small", "large"]:
            for i, (labels, features) in enumerate(datasets[size]["data"]):
                self.labels = labels
                self.features = features
                self.num_instances, self.num_features = features.shape
                self.current_dataset = datasets[size]["files"][i]

                summary[size].append(self.get_stats())

        return summary


def main():
    loader = DataLoader()
    folder_path = "data/"

    try:
        datasets = loader.load_datasets(folder_path)

        summary = loader.get_dataset_summary(datasets)

        for size in ["small", "large"]:
            print(f"\n{size.upper()} Datasets:")
            for stats in summary[size]:
                print(f"\nDataset: {stats['dataset_name']}")
                print(f"Instances: {stats['num_instances']}")
                print(f"Features: {stats['num_features']}")
                print(f"Class distribution: {stats['class_distribution']}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
