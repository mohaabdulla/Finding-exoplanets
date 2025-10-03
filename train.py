"""
Training script for exoplanet classification using machine learning.

This script implements a two-stage approach:
1. Stage 1: IsolationForest for anomaly detection and filtering
2. Stage 2: Multiple classifiers (LogisticRegression, DecisionTree, RandomForest)

Uses Monte Carlo cross-validation for robust model evaluation.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib

warnings.filterwarnings("ignore")


class ExoplanetClassifier:
    """
    A two-stage exoplanet classification system.
    
    Stage 1: IsolationForest for anomaly detection
    Stage 2: Multiple classifiers for final prediction
    """

    def __init__(self, random_state=42):
        """
        Initialize the classifier.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []

        # Stage 1: Anomaly detection
        self.anomaly_detector = IsolationForest(
            contamination=0.1, random_state=random_state, n_jobs=-1
        )

        # Stage 2: Classifiers
        self.classifiers = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, random_state=random_state, n_jobs=-1
            ),
            "DecisionTree": DecisionTreeClassifier(random_state=random_state),
            "RandomForest": RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
        }

        self.best_classifier_name = None
        self.best_classifier = None

    def load_data(self, file_path):
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def prepare_target(self, df, target_column="koi_disposition"):
        """
        Prepare target variable (CONFIRMED=1, others=0).
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            DataFrame with processed target
        """
        print(f"\nPreparing target variable '{target_column}'...")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Create binary target: CONFIRMED=1, others=0
        df["target"] = (df[target_column] == "CONFIRMED").astype(int)

        print(f"Target distribution:")
        print(df["target"].value_counts())
        print(f"Class balance: {df['target'].mean():.2%} positive class")

        return df

    def preprocess_data(self, df, target_column="target", fit=True):
        """
        Preprocess data with cleaning, imputation, scaling, and one-hot encoding.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            fit: Whether to fit transformers (True for training, False for prediction)
            
        Returns:
            Preprocessed features (X) and target (y)
        """
        print("\nPreprocessing data...")

        # Separate features and target
        if target_column in df.columns:
            y = df[target_column].values
            X = df.drop(columns=[target_column, "koi_disposition"], errors="ignore")
        else:
            y = None
            X = df.drop(columns=["koi_disposition"], errors="ignore")

        # Remove columns with all missing values
        X = X.dropna(axis=1, how="all")

        # Identify categorical and numerical features
        if fit:
            self.categorical_features = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            self.numerical_features = X.select_dtypes(
                include=["number"]
            ).columns.tolist()
            print(
                f"Found {len(self.categorical_features)} categorical and {len(self.numerical_features)} numerical features"
            )

        # Handle categorical features with label encoding
        for col in self.categorical_features:
            if col in X.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    # Handle missing values before encoding
                    X[col] = X[col].fillna("missing")
                    X[col] = self.label_encoders[col].fit_transform(X[col])
                else:
                    # Handle missing values before encoding
                    X[col] = X[col].fillna("missing")
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    X[col] = X[col].apply(
                        lambda x: le.transform([x])[0]
                        if x in le.classes_
                        else -1
                    )

        # Impute missing values for numerical features
        if fit:
            X[self.numerical_features] = self.imputer.fit_transform(
                X[self.numerical_features]
            )
        else:
            X[self.numerical_features] = self.imputer.transform(
                X[self.numerical_features]
            )

        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()

        # Scale numerical features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        print(f"Preprocessed data shape: {X_scaled.shape}")

        return X_scaled, y

    def monte_carlo_cross_validation(self, X, y, n_iterations=10, test_size=0.2):
        """
        Perform Monte Carlo cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_iterations: Number of random train/test splits
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with average metrics for each classifier
        """
        print(f"\n{'=' * 80}")
        print(
            f"Starting Monte Carlo Cross-Validation ({n_iterations} iterations)..."
        )
        print(f"{'=' * 80}")

        results = {
            name: {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
            }
            for name in self.classifiers.keys()
        }

        for iteration in range(n_iterations):
            print(f"\nIteration {iteration + 1}/{n_iterations}")
            print("-" * 40)

            # Random train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state + iteration
            )

            # Train and evaluate each classifier
            for name, clf in self.classifiers.items():
                clf_copy = clf.__class__(**clf.get_params())
                clf_copy.fit(X_train, y_train)
                y_pred = clf_copy.predict(X_test)

                results[name]["accuracy"].append(accuracy_score(y_test, y_pred))
                results[name]["precision"].append(
                    precision_score(y_test, y_pred, zero_division=0)
                )
                results[name]["recall"].append(
                    recall_score(y_test, y_pred, zero_division=0)
                )
                results[name]["f1"].append(
                    f1_score(y_test, y_pred, zero_division=0)
                )

        # Calculate and display average results
        print(f"\n{'=' * 80}")
        print("Monte Carlo Cross-Validation Results (Average ± Std)")
        print(f"{'=' * 80}")

        for name, metrics in results.items():
            print(f"\n{name}:")
            for metric_name, values in metrics.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {metric_name.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")

        # Find best classifier based on F1 score
        best_f1 = 0
        best_name = None
        for name, metrics in results.items():
            mean_f1 = np.mean(metrics["f1"])
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_name = name

        print(f"\n{'=' * 80}")
        print(f"Best classifier: {best_name} (F1: {best_f1:.4f})")
        print(f"{'=' * 80}")

        return results, best_name

    def train(self, X, y):
        """
        Train the two-stage classification system.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        print(f"\n{'=' * 80}")
        print("Stage 1: Anomaly Detection with IsolationForest")
        print(f"{'=' * 80}")

        # Stage 1: Anomaly detection
        print("Fitting IsolationForest...")
        self.anomaly_detector.fit(X)
        anomaly_predictions = self.anomaly_detector.predict(X)

        # Filter out anomalies (-1 means anomaly, 1 means normal)
        normal_mask = anomaly_predictions == 1
        X_filtered = X[normal_mask]
        y_filtered = y[normal_mask]

        n_anomalies = np.sum(~normal_mask)
        print(f"Detected {n_anomalies} anomalies ({n_anomalies / len(X) * 100:.2f}%)")
        print(f"Training data after filtering: {X_filtered.shape[0]} samples")

        # Stage 2: Train classifiers with Monte Carlo cross-validation
        print(f"\n{'=' * 80}")
        print("Stage 2: Training Classifiers")
        print(f"{'=' * 80}")

        results, best_name = self.monte_carlo_cross_validation(
            X_filtered, y_filtered, n_iterations=10
        )

        # Train best classifier on all filtered data
        self.best_classifier_name = best_name
        self.best_classifier = self.classifiers[best_name]
        print(f"\nTraining final {best_name} model on all filtered data...")
        self.best_classifier.fit(X_filtered, y_filtered)
        print("Training complete!")

        return results

    def evaluate(self, X, y):
        """
        Evaluate the trained model.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        print(f"\n{'=' * 80}")
        print("Final Model Evaluation")
        print(f"{'=' * 80}")

        # Apply anomaly detection
        anomaly_predictions = self.anomaly_detector.predict(X)
        normal_mask = anomaly_predictions == 1
        X_filtered = X[normal_mask]
        y_filtered = y[normal_mask]

        # Make predictions
        y_pred = self.best_classifier.predict(X_filtered)

        # Calculate metrics
        accuracy = accuracy_score(y_filtered, y_pred)
        precision = precision_score(y_filtered, y_pred, zero_division=0)
        recall = recall_score(y_filtered, y_pred, zero_division=0)
        f1 = f1_score(y_filtered, y_pred, zero_division=0)

        print(f"\nBest Model: {self.best_classifier_name}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_filtered, y_pred, target_names=["Not Confirmed", "Confirmed"]))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_filtered, y_pred)
        print(cm)

    def save_models(self, output_dir="models"):
        """
        Save trained models and preprocessors.
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving models to '{output_dir}' directory...")

        # Save preprocessors
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        joblib.dump(self.imputer, os.path.join(output_dir, "imputer.pkl"))
        joblib.dump(
            self.label_encoders, os.path.join(output_dir, "label_encoders.pkl")
        )

        # Save feature information
        joblib.dump(
            {
                "feature_names": self.feature_names,
                "categorical_features": self.categorical_features,
                "numerical_features": self.numerical_features,
            },
            os.path.join(output_dir, "feature_info.pkl"),
        )

        # Save anomaly detector
        joblib.dump(
            self.anomaly_detector, os.path.join(output_dir, "anomaly_detector.pkl")
        )

        # Save best classifier
        joblib.dump(
            self.best_classifier, os.path.join(output_dir, "best_classifier.pkl")
        )
        joblib.dump(
            {"name": self.best_classifier_name},
            os.path.join(output_dir, "best_classifier_name.pkl"),
        )

        print("Models saved successfully!")

    def load_models(self, model_dir="models"):
        """
        Load trained models and preprocessors.
        
        Args:
            model_dir: Directory containing saved models
        """
        print(f"Loading models from '{model_dir}' directory...")

        # Load preprocessors
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.imputer = joblib.load(os.path.join(model_dir, "imputer.pkl"))
        self.label_encoders = joblib.load(
            os.path.join(model_dir, "label_encoders.pkl")
        )

        # Load feature information
        feature_info = joblib.load(os.path.join(model_dir, "feature_info.pkl"))
        self.feature_names = feature_info["feature_names"]
        self.categorical_features = feature_info["categorical_features"]
        self.numerical_features = feature_info["numerical_features"]

        # Load anomaly detector
        self.anomaly_detector = joblib.load(
            os.path.join(model_dir, "anomaly_detector.pkl")
        )

        # Load best classifier
        self.best_classifier = joblib.load(
            os.path.join(model_dir, "best_classifier.pkl")
        )
        self.best_classifier_name = joblib.load(
            os.path.join(model_dir, "best_classifier_name.pkl")
        )["name"]

        print(f"Models loaded successfully! Best classifier: {self.best_classifier_name}")


def main():
    """Main training function."""
    print("=" * 80)
    print("Exoplanet Classification Training Pipeline")
    print("=" * 80)

    # Configuration
    DATA_FILE = "cumulative_2025.10.03_09.12.20.csv"
    MODEL_DIR = "models"
    RANDOM_STATE = 42

    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"\nError: Data file '{DATA_FILE}' not found!")
        print("Please place the data file in the current directory.")
        return

    # Initialize classifier
    classifier = ExoplanetClassifier(random_state=RANDOM_STATE)

    # Load data
    df = classifier.load_data(DATA_FILE)

    # Prepare target
    df = classifier.prepare_target(df)

    # Preprocess data
    X, y = classifier.preprocess_data(df, fit=True)

    # Split data for training and final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train models
    results = classifier.train(X_train, y_train)

    # Evaluate on test set
    classifier.evaluate(X_test, y_test)

    # Save models
    classifier.save_models(MODEL_DIR)

    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
