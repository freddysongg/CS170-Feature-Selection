# Feature Selection with Nearest Neighbor

## Project Overview
This project implements feature selection algorithms using the nearest neighbor classifier. The implementation includes both Forward Selection and Backward Elimination approaches, with performance tracking, weak feature analysis, and visualization of results.

## Project Structure
```
CS170-Feature-Selection/
├── data/                      # Dataset directory
│   ├── CS170_Small_Data__112.txt
│   └── CS170_Large_Data__38.txt
├── metrics/                   # Performance metrics storage
│   ├── *_ForwardSelector_metrics.json
│   └── *_BackwardSelector_metrics.json
├── plots/                     # Generated visualization plots
│   ├── accuracy_vs_features/
│   └── runtime_comparison.png
├── src/                      
    ├── data_loader.py        # Data loading and validation
    ├── classifier.py         # Nearest Neighbor implementation
    ├── feature_selection.py  # Feature selection algorithms
    ├── metrics_collector.py  # Performance tracking
    └── visualize_results.py  # Results visualization
```

## Implementation Details

### 1. Data Loading (`data_loader.py`)
- Handles ASCII text file loading
- Validates data format and class labels (1 or 2)
- Supports multiple dataset loading
- Performs basic data validation and statistics

### 2. Nearest Neighbor Classifier (`classifier.py`)
- Implements 1-NN classification using Euclidean distance
- Supports feature subset evaluation
- Includes leave-one-out cross-validation (LOOCV)

### 3. Feature Selection (`feature_selection.py`)
- Base FeatureSelector class with shared functionality
- Forward Selection implementation
- Backward Elimination implementation
- Weak feature analysis

### 4. Performance Metrics (`metrics_collector.py`)
- Tracks comprehensive metrics including:
  - Accuracy history for each feature set
  - Runtime performance
  - Dataset statistics
  - Individual feature performance
  - Feature subset performance
  - Weak feature identification

### 5. Visualization (`visualize_results.py`)
- Generates accuracy vs. feature count plots
- Creates runtime comparison charts
- Visualizes weak feature analysis

## Feature Selection Algorithms

### Forward Selection
1. Starts with empty feature set
2. Iteratively adds best performing feature
3. Evaluates all possible features
4. Tracks accuracy history and runtime

### Backward Elimination
1. Starts with all features
2. Iteratively removes features
3. Evaluates remaining feature sets
4. Records performance metrics
5. Analyzes overshadowed features

### Weak Feature Analysis
- Identifies features that perform well individually but get overshadowed
- Calculates individual feature performance

## Usage

1. Install Requirements:
```bash
pip install numpy matplotlib seaborn
```

2. Run Feature Selection:
```bash
python src/feature_selection.py
```

3. Generate Visualizations:
```bash
python src/visualize_results.py
```

## Output Format

1. Console Output:
```
This dataset has X features (not including the class attribute), with Y instances.
Running nearest neighbor with all X features, using "leave-one-out" evaluation...
Beginning search.
Using feature(s) {1} accuracy is A%
...
Analyzing individual feature performance:
Feature 1 individual accuracy: X%
...
Weak features:
Feature N: X% accuracy individually
```

2. Performance Metrics (JSON):
```json
{
    "dataset_name": "example_dataset.txt",
    "algorithm": "ForwardSelector",
    "dataset_size": 500,
    "total_features": 6,
    "runtime": 19.13,
    "accuracy_history": [...],
    "best_accuracy": 0.978,
    "best_feature_count": 2,
    "best_features": [1, 3],
    "individual_feature_accuracies": {...},
    "weak_features": [...]
}
```

3. Visualization Plots:
- Accuracy progression graphs
- Runtime comparison charts
- Feature performance analysis

## Implementation Notes

1. Data Format Requirements:
   - ASCII text files
   - Space-separated values
   - First column: class labels (1 or 2)
   - Remaining columns: continuous features
   - Maximum 64 features
   - Maximum 2,048 instances

2. Performance Tracking:
   - Automated metrics collection
   - Runtime measurement
   - Weak feature identification

3. Analysis Features:
   - Individual feature performance
   - Feature interaction analysis
   - Performance visualization

## Testing

Run components independently:
```bash
# Test data loading
python src/data_loader.py

# Test classifier
python src/classifier.py

# Run feature selection
python src/feature_selection.py

# Generate visualizations
python src/visualize_results.py
```

## Author
Freddy Song

## License
MIT License