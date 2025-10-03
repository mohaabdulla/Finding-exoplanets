"""
Exoplanet classification with astrophysical-only features, engineered features,
and a Habitable Zone (HZ) flag. No fixed random_state (non-deterministic runs).

Outputs:
- Monte Carlo CV metrics
- Final evaluation
- HZ counts and percentages:
  (a) True CONFIRMED planets in HZ / total true CONFIRMED
  (b) Predicted planets in HZ / total predicted planets
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
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
import joblib

warnings.filterwarnings("ignore")


# ----------------------------- Feature engineering -----------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add general engineered features (logs, ratios, colors, duration/period)."""
    df = df.copy()

    # Log transforms (use log1p for safety)
    for col in ["koi_period", "koi_depth", "koi_prad", "koi_insol"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # Ratios / normalizations
    if {"koi_depth", "koi_srad"}.issubset(df.columns):
        df["depth_norm"] = df["koi_depth"] / df["koi_srad"]
    if {"koi_prad", "koi_srad"}.issubset(df.columns):
        df["prad_srad_ratio"] = df["koi_prad"] / df["koi_srad"]
    if {"koi_sma", "koi_srad"}.issubset(df.columns):
        df["a_scaled"] = df["koi_sma"] / df["koi_srad"]

    # Colors from magnitudes
    if {"koi_gmag", "koi_rmag"}.issubset(df.columns):
        df["g_r"] = df["koi_gmag"] - df["koi_rmag"]
    if {"koi_rmag", "koi_imag"}.issubset(df.columns):
        df["r_i"] = df["koi_rmag"] - df["koi_imag"]
    if {"koi_jmag", "koi_hmag"}.issubset(df.columns):
        df["j_h"] = df["koi_jmag"] - df["koi_hmag"]

    # Duration vs period
    if {"koi_duration", "koi_period"}.issubset(df.columns):
        df["dur_period_ratio"] = df["koi_duration"] / df["koi_period"]

    return df


def add_habitability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Habitable Zone fields using:
      L* ~ (R*/Rsun)^2 * (Teff/5777)^4
      d_center = sqrt(L*)
      d_inner = 0.95 * d_center
      d_outer = 1.37 * d_center
    habitable_zone = 1 if d_inner <= a (koi_sma) <= d_outer else 0
    """
    df = df.copy()
    needed = {"koi_srad", "koi_steff", "koi_sma"}
    if needed.issubset(df.columns):
        stellar_lum = (df["koi_srad"] ** 2) * (df["koi_steff"] / 5777.0) ** 4
        d_center = np.sqrt(stellar_lum)
        d_inner = 0.95 * d_center
        d_outer = 1.37 * d_center

        df["stellar_lum"] = stellar_lum
        df["d_center"] = d_center
        df["d_inner"] = d_inner
        df["d_outer"] = d_outer
        df["habitable_zone"] = (
            (df["koi_sma"] >= d_inner) & (df["koi_sma"] <= d_outer)
        ).astype(int)
    else:
        # If any required column is missing, create a safe default
        df["stellar_lum"] = np.nan
        df["d_center"] = np.nan
        df["d_inner"] = np.nan
        df["d_outer"] = np.nan
        df["habitable_zone"] = 0

    return df


# ------------------------------- Classifier ------------------------------------

class ExoplanetClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []

        # Stage 1: anomaly detection (random_state=None for stochastic runs)
        self.anomaly_detector = IsolationForest(
            contamination=0.10, random_state=None, n_jobs=-1
        )

        # Stage 2: classifiers
        self.classifiers = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, random_state=None, n_jobs=-1
            ),
            "DecisionTree": DecisionTreeClassifier(random_state=None),
            "RandomForest": RandomForestClassifier(
                n_estimators=300, random_state=None, n_jobs=-1
            ),
        }

        self.best_classifier_name = None
        self.best_classifier = None

    # --------------------------- Data handling ---------------------------

    def load_data(self, file_path):
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def prepare_target(self, df, target_column="koi_disposition"):
        print(f"\nPreparing target variable '{target_column}'...")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        df = df.copy()
        df["target"] = (df[target_column] == "CONFIRMED").astype(int)
        print(df["target"].value_counts())
        print(f"Class balance: {df['target'].mean():.2%} positive class")
        return df

    def select_safe_features(self, df):
        """Whitelist astrophysical/stellar/magnitudes + engineered + HZ fields."""
        base = [
            "koi_period", "koi_duration", "koi_depth", "koi_ror",
            "koi_srho", "koi_prad", "koi_sma", "koi_incl",
            "koi_teq", "koi_insol", "koi_dor",
            "koi_steff", "koi_slogg", "koi_smet",
            "koi_srad", "koi_smass",
            "ra", "dec",
            "koi_kepmag", "koi_gmag", "koi_rmag", "koi_imag", "koi_zmag",
            "koi_jmag", "koi_hmag", "koi_kmag",
            # engineered
            "log_koi_period", "log_koi_depth", "log_koi_prad", "log_koi_insol",
            "depth_norm", "prad_srad_ratio", "a_scaled",
            "g_r", "r_i", "j_h", "dur_period_ratio",
            # habitability
            "stellar_lum", "d_center", "d_inner", "d_outer", "habitable_zone",
        ]
        keep = [c for c in base if c in df.columns]
        cols = keep + ["target"]
        return df[cols]

    def preprocess_data(self, df, target_column="target", fit=True):
        """
        Returns:
            X_scaled: np.ndarray
            y: np.ndarray or None
            hz: pd.Series aligned with X (habitable_zone flag)
        """
        print("\nPreprocessing data...")

        # Engineering + HZ
        df = add_engineered_features(df)
        df = add_habitability(df)

        # Restrict to safe features
        df = self.select_safe_features(df)

        # Separate target
        if target_column in df.columns:
            y = df[target_column].values
            feats = df.drop(columns=[target_column])
        else:
            y = None
            feats = df.copy()

        # Keep HZ flag for reporting
        hz = feats["habitable_zone"].fillna(0).astype(int) if "habitable_zone" in feats.columns else pd.Series(np.zeros(len(feats), dtype=int))
        # Remove columns with all NaN
        feats = feats.dropna(axis=1, how="all")

        # Type splits
        if fit:
            self.categorical_features = feats.select_dtypes(include=["object", "category"]).columns.tolist()
            self.numerical_features = feats.select_dtypes(include=["number"]).columns.tolist()
            print(f"Using {len(self.numerical_features)} numerical and {len(self.categorical_features)} categorical features")

        # Encode categoricals if any
        X = feats.copy()
        for col in self.categorical_features:
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna("missing"))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                X[col] = X[col].fillna("missing").apply(lambda v: le.transform([v])[0] if v in le.classes_ else -1)

        # Impute numerics
        if fit:
            X[self.numerical_features] = self.imputer.fit_transform(X[self.numerical_features])
        else:
            X[self.numerical_features] = self.imputer.transform(X[self.numerical_features])

        if fit:
            self.feature_names = X.columns.tolist()

        # Scale
        X_scaled = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)

        print(f"Preprocessed shape: {X_scaled.shape}")
        return X_scaled, y, hz.reset_index(drop=True)

    # --------------------------- Training / CV ---------------------------

    def monte_carlo_cv(self, X, y, n_iterations=10, test_size=0.8):
        print(f"\n{'='*80}\nMonte Carlo Cross-Validation ({n_iterations} iterations)\n{'='*80}")
        results = {name: {"accuracy": [], "precision": [], "recall": [], "f1": []} for name in self.classifiers.keys()}

        for i in range(n_iterations):
            print(f"\nIteration {i+1}/{n_iterations}")
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=None)
            for name, clf in self.classifiers.items():
                c = clf.__class__(**clf.get_params())
                c.fit(Xtr, ytr)
                yp = c.predict(Xte)
                results[name]["accuracy"].append(accuracy_score(yte, yp))
                results[name]["precision"].append(precision_score(yte, yp, zero_division=0))
                results[name]["recall"].append(recall_score(yte, yp, zero_division=0))
                results[name]["f1"].append(f1_score(yte, yp, zero_division=0))

        best_f1, best_name = -1, None
        print(f"\n{'='*80}\nMC-CV Results (Average ± Std)\n{'='*80}")
        for name, m in results.items():
            acc = (np.mean(m["accuracy"]), np.std(m["accuracy"]))
            pre = (np.mean(m["precision"]), np.std(m["precision"]))
            rec = (np.mean(m["recall"]), np.std(m["recall"]))
            f1  = (np.mean(m["f1"]), np.std(m["f1"]))
            print(f"\n{name}:\n  Accuracy:  {acc[0]:.4f} ± {acc[1]:.4f}\n  Precision: {pre[0]:.4f} ± {pre[1]:.4f}\n  Recall:    {rec[0]:.4f} ± {rec[1]:.4f}\n  F1:        {f1[0]:.4f} ± {f1[1]:.4f}")
            if f1[0] > best_f1:
                best_f1, best_name = f1[0], name

        print(f"\nBest classifier: {best_name} (F1 {best_f1:.4f})")
        return results, best_name

    def train(self, X, y):
        print(f"\n{'='*80}\nStage 1: IsolationForest\n{'='*80}")
        self.anomaly_detector.fit(X)
        mask = self.anomaly_detector.predict(X) == 1
        Xf, yf = X[mask], y[mask]
        print(f"Training data after filtering: {Xf.shape[0]} samples")

        print(f"\n{'='*80}\nStage 2: Classifiers\n{'='*80}")
        _, best_name = self.monte_carlo_cv(Xf, yf)
        self.best_classifier_name = best_name
        self.best_classifier = self.classifiers[best_name]
        self.best_classifier.fit(Xf, yf)
        print(f"Final model: {best_name}")

    def evaluate(self, X, y, hz_flag_test: pd.Series):
        print(f"\n{'='*80}\nFinal Evaluation\n{'='*80}")
        mask = self.anomaly_detector.predict(X) == 1
        Xf, yf = X[mask], y[mask]
        hz_f = hz_flag_test.iloc[mask.nonzero()[0]].reset_index(drop=True)

        yp = self.best_classifier.predict(Xf)
        print(classification_report(yf, yp, target_names=["Not Confirmed", "Confirmed"]))
        print("Confusion Matrix:\n", confusion_matrix(yf, yp))

        # ----------------- HZ stats -----------------
        # True HZ among true planets
        true_planets = (yf == 1)
        n_true_planets = int(true_planets.sum())
        n_true_hz_planets = int((true_planets & (hz_f == 1)).sum())

        # Predicted HZ among predicted planets
        pred_planets = (yp == 1)
        n_pred_planets = int(pred_planets.sum())
        n_pred_hz_planets = int((pred_planets & (hz_f == 1)).sum())

        print("\n" + "-"*80)
        print("Habitable Zone Summary (test, after anomaly filtering)")
        print("-"*80)
        print(f"True CONFIRMED planets: {n_true_planets}")
        print(f"  In HZ (true): {n_true_hz_planets}  -> { (n_true_hz_planets / n_true_planets * 100) if n_true_planets else 0:.2f}%")
        print(f"Predicted planets:      {n_pred_planets}")
        print(f"  In HZ (pred):  {n_pred_hz_planets}  -> { (n_pred_hz_planets / n_pred_planets * 100) if n_pred_planets else 0:.2f}%")

        # Return counts for programmatic use if needed
        return {
            "true_planets": n_true_planets,
            "true_hz_planets": n_true_hz_planets,
            "pred_planets": n_pred_planets,
            "pred_hz_planets": n_pred_hz_planets,
        }

    def save_models(self, outdir="models"):
        os.makedirs(outdir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(outdir, "scaler.pkl"))
        joblib.dump(self.imputer, os.path.join(outdir, "imputer.pkl"))
        joblib.dump(self.label_encoders, os.path.join(outdir, "label_encoders.pkl"))
        joblib.dump(self.anomaly_detector, os.path.join(outdir, "anomaly.pkl"))
        joblib.dump(self.best_classifier, os.path.join(outdir, "clf.pkl"))
        joblib.dump({"name": self.best_classifier_name}, os.path.join(outdir, "clf_name.pkl"))
        print("Models saved to", outdir)


# ---------------------------------- Main ---------------------------------------

def main():
    print("="*80)
    print("Exoplanet Classification Training Pipeline (Astrophysical Features + HZ)")
    print("="*80)

    DATA_FILE = "cumulative_2025.10.03_09.12.20.csv"
    MODEL_DIR = "models"

    if not os.path.exists(DATA_FILE):
        print("Error: data file not found.")
        return

    clf = ExoplanetClassifier()
    df = clf.load_data(DATA_FILE)
    df = clf.prepare_target(df)

    # Full-dataset HZ ratio on ground truth (for reference)
    df_hz_view = add_habitability(df)
    total_true_planets = int((df_hz_view["koi_disposition"] == "CONFIRMED").sum())
    total_true_hz = int(((df_hz_view["koi_disposition"] == "CONFIRMED") & (df_hz_view["habitable_zone"] == 1)).sum())
    print("\n--- Global HZ among true CONFIRMED (full dataset) ---")
    print(f"Total true planets: {total_true_planets}")
    print(f"In HZ (true):       {total_true_hz}  -> {(total_true_hz / total_true_planets * 100) if total_true_planets else 0:.2f}%")

    # Preprocess
    X, y, hz = clf.preprocess_data(df, fit=True)

    # Split
    Xtr, Xte, ytr, yte, hz_tr, hz_te = train_test_split(
        X, y, hz, test_size=0.8, stratify=y, random_state=None
    )
    print(f"\nTraining set {Xtr.shape[0]}, Test set {Xte.shape[0]}")

    # Train + eval
    clf.train(Xtr, ytr)
    hz_counts = clf.evaluate(Xte, yte, hz_te)

    # Save
    clf.save_models(MODEL_DIR)
    print("="*80, "\nTraining complete!")

    # Print compact return-like summary
    print("\nSummary counts (test, after anomaly filtering):")
    print(hz_counts)


if __name__ == "__main__":
    main()
