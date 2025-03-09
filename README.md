# Feature Selection with Nearest Neighbor

## Project Overview
This project implements feature selection algorithms using the nearest neighbor classifier. The implementation includes both Forward Selection and Backward Elimination approaches to find the optimal feature subset for classification.

## Project Structure
```
CS170-Feature-Selection/
├── data/                  # Dataset directory
│   ├── CS170_Small_Data112.txt
│   └── CS170_Large_Data38.txt
├── src/                   # Source code
│   ├── data_loader.py     # Data loading and validation
│   ├── classifier.py      # Nearest Neighbor implementation
│   └── feature_selection.py # Feature selection algorithms
└── README.md              # Project documentation
```

## Implementation Details

### 1. Data Loading (`data_loader.py`)
- Handles ASCII text file loading
- Validates data format and class labels (1 or 2)
- Supports multiple dataset loading
- Performs basic data validation and statistics

### 2. Nearest Neighbor Classifier (`classifier.py`)
- Implements 1-NN classification
- Uses Euclidean distance metric
- Supports feature subset evaluation
- Includes leave-one-out cross-validation

### 3. Feature Selection (`feature_selection.py`)
- Base `FeatureSelector` class for common functionality
- Forward Selection implementation
- Backward Elimination implementation
- Progress tracking and result reporting

## Algorithms

### Forward Selection
1. Starts with an empty feature set
2. Iteratively adds the best-performing feature
3. Continues until no improvement or all features are added
4. Returns the best feature subset found

### Backward Elimination
1. Starts with all features
2. Iteratively removes the least useful feature
3. Continues until no improvement or only one feature remains
4. Returns the best feature subset found

## Usage

### 1. Install Requirements:
```bash
pip install numpy
```

### 2. Place Datasets:
- Put your data files in the `data/` directory
- Files should be in ASCII format with space-separated values
- First column must be class labels (1 or 2)
- Remaining columns are feature values

### 3. Run Feature Selection:
```bash
python src/feature_selection.py
```

## Output Format
The program provides detailed output showing:

```
This dataset has X features (not including the class attribute), with Y instances.
Running nearest neighbor with all X features, using "leave-one-out" evaluation, I get an accuracy of Z%
Beginning search.
Using feature(s) {1} accuracy is A%
Using feature(s) {2} accuracy is B%
...
Feature set {2} was best, accuracy is C%
...
Finished search!! The best feature subset is {2,4}, which has an accuracy of D%
```

## Implementation Notes

### 1. Data Format Requirements:
- ASCII text files
- Space-separated values
- First column: class labels (1 or 2)
- Remaining columns: continuous features
- No missing values allowed
- Maximum 64 features
- Maximum 2,048 instances

### 2. Performance Considerations:
- Efficient nearest neighbor implementation
- Early stopping when no improvement
- Progress tracking for long runs

### 3. Validation:
- Input data validation
- Feature subset validation
- Cross-validation for accuracy

## Testing

Each component can be tested independently:

```bash
# Test data loading
python src/data_loader.py

# Test classifier
python src/classifier.py

# Run full feature selection
python src/feature_selection.py
```

## Results Analysis

The program provides:
1. Accuracy for each feature subset
2. Best feature subset found
3. Search path taken
4. Comparison between forward and backward selection

## Limitations
- Only handles binary classification (class 1 or 2)
- Requires continuous feature values
- No handling of missing values
- Limited to specified maximum features/instances

## Author
[Your Name]

## License
MIT License