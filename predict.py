"""
Prediction script for exoplanet classification.

This script loads trained models and makes predictions on new data.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from train import ExoplanetClassifier


def load_and_predict(data_file, model_dir="models", output_file=None):
    """
    Load trained models and make predictions.
    
    Args:
        data_file: Path to CSV file with data to predict
        model_dir: Directory containing saved models
        output_file: Optional path to save predictions (CSV format)
        
    Returns:
        DataFrame with predictions
    """
    print("=" * 80)
    print("Exoplanet Classification - Prediction")
    print("=" * 80)

    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"\nError: Model directory '{model_dir}' not found!")
        print("Please train the model first using train.py")
        return None

    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"\nError: Data file '{data_file}' not found!")
        return None

    # Initialize classifier and load models
    classifier = ExoplanetClassifier()
    classifier.load_models(model_dir)

    # Load data
    print(f"\nLoading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Keep original data for output
    df_original = df.copy()

    # Preprocess data (without target)
    X, _ = classifier.preprocess_data(df, target_column=None, fit=False)

    # Apply anomaly detection
    print("\nApplying anomaly detection...")
    anomaly_predictions = classifier.anomaly_detector.predict(X)
    is_normal = anomaly_predictions == 1

    # Make predictions
    print(f"Making predictions using {classifier.best_classifier_name}...")
    predictions = np.zeros(len(X), dtype=int)
    predictions[is_normal] = classifier.best_classifier.predict(X[is_normal])

    # Get prediction probabilities (if available)
    if hasattr(classifier.best_classifier, "predict_proba"):
        probabilities = np.zeros((len(X), 2))
        probabilities[is_normal] = classifier.best_classifier.predict_proba(
            X[is_normal]
        )
        confidence = probabilities[:, 1]  # Probability of positive class
    else:
        confidence = None

    # Add predictions to dataframe
    df_original["prediction"] = predictions
    df_original["prediction_label"] = df_original["prediction"].map(
        {0: "NOT_CONFIRMED", 1: "CONFIRMED"}
    )
    df_original["is_anomaly"] = ~is_normal

    if confidence is not None:
        df_original["confidence"] = confidence

    # Display prediction summary
    print("\n" + "=" * 80)
    print("Prediction Summary")
    print("=" * 80)
    print(f"\nTotal samples: {len(df_original)}")
    print(f"Anomalies detected: {np.sum(~is_normal)} ({np.sum(~is_normal) / len(df_original) * 100:.2f}%)")
    print(
        f"CONFIRMED predictions: {np.sum(predictions)} ({np.sum(predictions) / len(predictions) * 100:.2f}%)"
    )
    print(
        f"NOT_CONFIRMED predictions: {len(predictions) - np.sum(predictions)} ({(len(predictions) - np.sum(predictions)) / len(predictions) * 100:.2f}%)"
    )

    # Save predictions if output file specified
    if output_file:
        print(f"\nSaving predictions to {output_file}...")
        df_original.to_csv(output_file, index=False)
        print("Predictions saved successfully!")

    # Display first few predictions
    print("\nFirst 10 predictions:")
    columns_to_show = ["prediction_label", "is_anomaly"]
    if confidence is not None:
        columns_to_show.append("confidence")
    if "koi_disposition" in df_original.columns:
        columns_to_show.insert(0, "koi_disposition")

    print(df_original[columns_to_show].head(10).to_string())

    return df_original


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description="Predict exoplanet classifications using trained models"
    )
    parser.add_argument(
        "data_file",
        type=str,
        help="Path to CSV file with data to predict",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained models (default: models)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save predictions (CSV format)",
    )

    args = parser.parse_args()

    # Make predictions
    df_predictions = load_and_predict(
        args.data_file, args.model_dir, args.output
    )

    if df_predictions is None:
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Prediction Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
