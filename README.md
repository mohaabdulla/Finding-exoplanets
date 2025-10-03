# Finding Exoplanets - Machine Learning Classification

A machine learning project for classifying exoplanets using NASA's Kepler Object of Interest (KOI) dataset. This project implements a two-stage classification approach with anomaly detection and multiple classifiers, validated using Monte Carlo cross-validation.

## Overview

This project classifies exoplanets as **CONFIRMED** or **NOT CONFIRMED** based on various astronomical measurements from the Kepler mission. The classification pipeline consists of:

### Two-Stage Approach

1. **Stage 1: Anomaly Detection**
   - Uses IsolationForest to identify and filter out anomalies in the data
   - Helps improve model performance by removing outliers

2. **Stage 2: Classification**
   - Trains multiple classifiers:
     - Logistic Regression
     - Decision Tree
     - Random Forest
   - Selects the best performer based on F1 score

### Key Features

- **Data Preprocessing**: Comprehensive data cleaning, missing value imputation, feature scaling, and encoding
- **Monte Carlo Cross-Validation**: Robust model evaluation with multiple random train/test splits
- **Model Persistence**: Save and load trained models for future predictions
- **Anomaly Filtering**: Automatic detection and removal of data anomalies
- **Multiple Classifiers**: Compares several algorithms to find the best performer

## Dataset

The project uses the NASA Kepler KOI dataset: `cumulative_2025.10.03_09.12.20.csv`

- **Target Variable**: `koi_disposition`
  - `CONFIRMED` → 1 (positive class)
  - All other values → 0 (negative class)

The dataset can be downloaded from [NASA Exoplanet Archive](https://exoplanetarchive.ipsl.nasa.gov/).

## Project Structure

```
Finding-exoplanets/
├── train.py                  # Training script with full pipeline
├── predict.py                # Prediction script for new data
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── cumulative_2025.10.03_09.12.20.csv  # Dataset (not tracked in git)
└── models/                   # Saved models directory (created after training)
    ├── scaler.pkl
    ├── imputer.pkl
    ├── label_encoders.pkl
    ├── feature_info.pkl
    ├── anomaly_detector.pkl
    ├── best_classifier.pkl
    └── best_classifier_name.pkl
```

## Setup

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mohaabdulla/Finding-exoplanets.git
cd Finding-exoplanets
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Download `cumulative_2025.10.03_09.12.20.csv` from the NASA Exoplanet Archive
   - Place it in the project root directory

## Usage

### Training the Model

To train the classification models:

```bash
python train.py
```

This will:
1. Load and preprocess the data
2. Apply anomaly detection using IsolationForest
3. Train multiple classifiers with Monte Carlo cross-validation (10 iterations)
4. Select and train the best classifier on all data
5. Evaluate the final model on a held-out test set
6. Save all trained models to the `models/` directory

**Expected Output:**
- Data loading and preprocessing information
- Anomaly detection statistics
- Monte Carlo cross-validation results for each classifier
- Final model evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Classification Report
  - Confusion Matrix

### Making Predictions

To make predictions on new data:

```bash
python predict.py <data_file> [--model-dir models] [--output predictions.csv]
```

**Arguments:**
- `data_file`: Path to CSV file with data to predict (required)
- `--model-dir`: Directory containing trained models (default: `models`)
- `--output`: Output file to save predictions (optional)

**Example:**
```bash
# Predict on the same dataset
python predict.py cumulative_2025.10.03_09.12.20.csv --output predictions.csv

# Predict on new data with custom model directory
python predict.py new_data.csv --model-dir ./my_models --output results.csv
```

**Output:**
The predictions file will include:
- All original columns from the input data
- `prediction`: Binary prediction (0 or 1)
- `prediction_label`: Text label (NOT_CONFIRMED or CONFIRMED)
- `is_anomaly`: Whether the sample was flagged as an anomaly
- `confidence`: Prediction confidence score (if available)

## Model Details

### Preprocessing Pipeline

1. **Data Cleaning**:
   - Remove columns with all missing values
   - Identify categorical and numerical features

2. **Categorical Encoding**:
   - Label encoding for categorical variables
   - Handle missing values with "missing" category

3. **Numerical Imputation**:
   - Median imputation for missing numerical values

4. **Feature Scaling**:
   - StandardScaler for numerical features (zero mean, unit variance)

### Stage 1: Anomaly Detection

- **Algorithm**: IsolationForest
- **Contamination**: 0.1 (10% of data expected as anomalies)
- **Purpose**: Filter out outliers before classification

### Stage 2: Classification

Three classifiers are trained and compared:

1. **Logistic Regression**
   - Linear classifier
   - Fast training and prediction
   - Good for linearly separable data

2. **Decision Tree**
   - Non-linear classifier
   - Captures complex patterns
   - Interpretable decision rules

3. **Random Forest**
   - Ensemble of decision trees
   - Robust to overfitting
   - Handles non-linear relationships well

### Evaluation Strategy

**Monte Carlo Cross-Validation** with 10 iterations:
- Each iteration uses a different random 80/20 train/test split
- Calculates mean and standard deviation of metrics
- Provides robust performance estimates
- Selects best classifier based on F1 score

## Evaluation Metrics

The model is evaluated using multiple metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## Development

### Running Tests

The project includes basic validation in the training script. To verify everything works:

```bash
python train.py
```

Ensure the output shows:
- Successful data loading
- Preprocessing statistics
- Monte Carlo cross-validation results
- Final evaluation metrics
- Model saving confirmation

### Code Structure

The main class `ExoplanetClassifier` in `train.py` provides:

- `load_data()`: Load CSV data
- `prepare_target()`: Create binary target variable
- `preprocess_data()`: Full preprocessing pipeline
- `monte_carlo_cross_validation()`: Evaluation strategy
- `train()`: Two-stage training process
- `evaluate()`: Model evaluation
- `save_models()`: Save trained models
- `load_models()`: Load trained models

## Troubleshooting

### Common Issues

1. **"Data file not found" error**:
   - Ensure `cumulative_2025.10.03_09.12.20.csv` is in the project directory
   - Check file name spelling

2. **"Model directory not found" error**:
   - Train the model first using `python train.py`
   - Ensure the `models/` directory was created

3. **Memory errors**:
   - The dataset is large; ensure sufficient RAM (>4GB recommended)
   - Consider reducing `n_iterations` in Monte Carlo CV

4. **Import errors**:
   - Reinstall dependencies: `pip install -r requirements.txt`
   - Ensure Python version is 3.7 or higher

## Results

After training, you should expect:

- **Anomaly Detection**: ~10% of data flagged as anomalies
- **Model Performance**: Varies based on data quality, typically:
  - Accuracy: 85-95%
  - F1-Score: 80-90%
- **Best Classifier**: Often Random Forest, but varies with data

## Dependencies

- `numpy>=1.24.0`: Numerical computing
- `pandas>=2.0.0`: Data manipulation
- `scikit-learn>=1.3.0`: Machine learning algorithms
- `joblib>=1.3.0`: Model serialization

## License

This project is open source and available for educational purposes.

## Acknowledgments

- NASA Kepler Mission for the exoplanet dataset
- scikit-learn community for machine learning tools

## Contact

For questions or issues, please open an issue on GitHub.