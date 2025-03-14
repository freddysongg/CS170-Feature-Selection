import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import pandas as pd


def load_metrics(metrics_dir="metrics"):
    metrics_files = glob(os.path.join(metrics_dir, "*.json"))
    all_metrics = []

    for file in metrics_files:
        with open(file, "r") as f:
            metrics = json.load(f)
            all_metrics.append(metrics)

    return all_metrics


def plot_accuracy_vs_features(metrics, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        if "accuracy_history" not in metric:
            print(
                f"Skipping accuracy plot for {metric['dataset_name']}: No accuracy history found"
            )
            continue

        plt.figure(figsize=(10, 6))

        # Handle both old and new format
        feature_counts = []
        accuracies = []
        for entry in metric["accuracy_history"]:
            if isinstance(entry, dict):  # New format
                feature_count = entry["feature_count"]
                # Skip the empty feature set (0 features)
                if feature_count > 0:
                    feature_counts.append(feature_count)
                    accuracies.append(entry["accuracy"])
            else:  # Old format
                feature_count = len(entry[0])
                # Skip the empty feature set (0 features)
                if feature_count > 0:
                    feature_counts.append(feature_count)
                    accuracies.append(entry[1])

        plt.plot(feature_counts, accuracies, marker="o", label="Feature Selection Path")

        # Get algorithm name (handle both keys)
        algorithm_name = metric.get(
            "algorithm", metric.get("algorithm_name", "Unknown")
        )

        plt.title(
            f'Accuracy vs Feature Count\n{metric["dataset_name"]} - {algorithm_name}'
        )
        plt.xlabel("Number of Features")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)

        plt.savefig(
            os.path.join(
                output_dir,
                f'{metric["dataset_name"]}_{algorithm_name}_accuracy.png',
            )
        )
        plt.close()


def plot_runtime_comparison(metrics, output_dir="plots"):
    plt.figure(figsize=(12, 6))

    data = {"Dataset": [], "Runtime (s)": [], "Algorithm": []}

    for metric in metrics:
        data["Dataset"].append(
            f'{metric["dataset_name"]}\n({metric["dataset_size"]} instances)'
        )
        data["Runtime (s)"].append(metric["runtime"])
        data["Algorithm"].append(metric["algorithm"])

    sns.barplot(x="Dataset", y="Runtime (s)", hue="Algorithm", data=pd.DataFrame(data))

    plt.title("Runtime Comparison by Dataset Size")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "runtime_comparison.png"))
    plt.close()


def plot_feature_comparison(metrics, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(12, 6))

        features_data = {"Feature": [], "Accuracy": [], "Category": []}

        # Get individual accuracies from accuracy history
        if "accuracy_history" in metric:
            for entry in metric["accuracy_history"]:
                if isinstance(entry, dict):  # New format
                    # Adjust feature count to start from 1
                    feature_count = max(1, entry["feature_count"])
                    accuracy = entry["accuracy"]
                else:  # Old format (list of features and accuracy)
                    # Adjust feature count to start from 1
                    feature_count = max(1, len(entry[0]))
                    accuracy = entry[1]

                features_data["Feature"].append(f"F{feature_count}")
                features_data["Accuracy"].append(accuracy * 100)
                features_data["Category"].append("Selected Features")

        # Add weak features if available
        if "weak_features" in metric:
            for feature, accuracy in metric["weak_features"]:
                # Adjust feature index to start from 1
                features_data["Feature"].append(f"F{feature + 1}")
                features_data["Accuracy"].append(accuracy * 100)
                features_data["Category"].append("Weak Features")

        if not features_data["Feature"]:  # Skip if no data
            print(
                f"Skipping visualization for {metric['dataset_name']}: No feature data found"
            )
            plt.close()
            continue

        df = pd.DataFrame(features_data)

        # Create grouped bar plot
        sns.barplot(
            data=df,
            x="Feature",
            y="Accuracy",
            hue="Category",
            palette=["blue", "orange"],
        )

        # Get algorithm name (handle both keys)
        algorithm_name = metric.get(
            "algorithm", metric.get("algorithm_name", "Unknown")
        )

        plt.title(f'Feature Comparison\n{metric["dataset_name"]} - {algorithm_name}')
        plt.xlabel("Features")
        plt.ylabel("Accuracy (%)")

        # Add best accuracy line if available
        if "best_accuracy" in metric:
            plt.axhline(
                y=metric["best_accuracy"] * 100,
                color="red",
                linestyle="--",
                label=f'Best Accuracy ({metric["best_accuracy"]*100:.1f}%)',
            )

        plt.legend(title="Feature Category")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                f'{metric["dataset_name"]}_{algorithm_name}_feature_comparison.png',
            )
        )
        plt.close()


def main():
    metrics = load_metrics()

    plot_accuracy_vs_features(metrics)
    plot_runtime_comparison(metrics)
    plot_feature_comparison(metrics)


if __name__ == "__main__":
    main()
