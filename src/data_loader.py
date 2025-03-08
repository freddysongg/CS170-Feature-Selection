import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import re


class DataLoader:
    def __init__(self):
        """Initialize the DataLoader class"""
        self.labels = None
        self.features = None
        self.num_features = 0
        self.num_instances = 0
        self.current_dataset = None

    def load_data(self, filename):
        """
        Loads and validates the dataset from a given filename.

        Args:
            filename (str): Path to the data file

        Returns:
            tuple: (labels, features) where labels are class identifiers and
                  features are the corresponding feature values

        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data format is invalid
        """
        try:
            if not Path(filename).exists():
                raise FileNotFoundError(f"Data file not found: {filename}")

            data = np.loadtxt(filename)

            if data.shape[1] < 2:
                raise ValueError(
                    "Data must have at least one feature and a class label"
                )

            self.labels = data[:, 0]
            self.features = data[:, 1:]

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
        """
        Returns basic statistics about the loaded dataset.

        Returns:
            dict: Dictionary containing dataset statistics
        """
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
        """
        Loads datasets from the specified folder, optionally filtering by type.

        Args:
            folder_path (str): Path to the data folder
            dataset_type (str, optional): 'small' or 'large' to filter datasets.
                                        If None, loads all datasets.

        Returns:
            dict: Dictionary containing dataset information with structure:
                {
                    'small': {
                        'files': list of small dataset files,
                        'data': list of (labels, features) tuples
                    },
                    'large': {
                        'files': list of large dataset files,
                        'data': list of (labels, features) tuples
                    }
                }
        """
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
        """
        Generates a summary of all loaded datasets.

        Args:
            datasets (dict): The datasets dictionary returned by load_datasets

        Returns:
            dict: Summary statistics for all datasets
        """
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
    """Example usage of the DataLoader class with full data directory"""
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
