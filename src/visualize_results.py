import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob


def load_metrics(metrics_dir="metrics"):
    metrics_files = glob(os.path.join(metrics_dir, "*.json"))
    all_metrics = []
    
    for file in metrics_files:
        with open(file, 'r') as f:
            metrics = json.load(f)
            all_metrics.append(metrics)
    
    return all_metrics

def plot_accuracy_vs_features(metrics, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        feature_counts = [h["feature_count"] for h in metric["accuracy_history"]]
        accuracies = [h["accuracy"] for h in metric["accuracy_history"]]
        
        plt.plot(feature_counts, accuracies, marker='o')
        plt.title(f'Accuracy vs Feature Count\n{metric["dataset_name"]} - {metric["algorithm"]}')
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        plt.savefig(os.path.join(
            output_dir, 
            f'{metric["dataset_name"]}_{metric["algorithm"]}_accuracy.png'
        ))
        plt.close()

def plot_runtime_comparison(metrics, output_dir="plots"):
    plt.figure(figsize=(12, 6))
    
    datasets = set(m["dataset_name"] for m in metrics)
    algorithms = set(m["algorithm"] for m in metrics)
    
    data = {
        "Dataset": [],
        "Runtime (s)": [],
        "Algorithm": []
    }
    
    for metric in metrics:
        data["Dataset"].append(f'{metric["dataset_name"]}\n({metric["dataset_size"]} instances)')
        data["Runtime (s)"].append(metric["runtime"])
        data["Algorithm"].append(metric["algorithm"])
    
    sns.barplot(
        x="Dataset",
        y="Runtime (s)",
        hue="Algorithm",
        data=data
    )
    
    plt.title('Runtime Comparison by Dataset Size')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'runtime_comparison.png'))
    plt.close()

def main():
    metrics = load_metrics()
    
    plot_accuracy_vs_features(metrics)
    plot_runtime_comparison(metrics)

if __name__ == "__main__":
    main() 