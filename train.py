"""
Exoplanet classification with:
- Safe astrophysical features only
- Feature engineering + Habitable Zone (HZ) flag
- IsolationForest anomaly filter
- Optional SMOTE oversampling for imbalance
- Monte Carlo CV (default 60 iterations) with PR-AUC + F1
- Hyperparameter tuning (RandomizedSearchCV, default 200 trials)
- Optional probability calibration (isotonic)
- Threshold tuning to maximize F1
- Optional soft-voting ensemble
- Prints ONLY planet names that are in the HZ
- FAST mode to reduce runtime

Run examples:
  python train.py --tune --mc-iters 100 --contamination 0.06 --rf-trees 800 --smote --calibrate --ensemble
  python train.py --fast
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import IsolationForest, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import randint, loguniform
import joblib

warnings.filterwarnings("ignore")

# ----------------------------- Feature engineering -----------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["koi_period", "koi_depth", "koi_prad", "koi_insol"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    if {"koi_depth", "koi_srad"}.issubset(df.columns):
        df["depth_norm"] = df["koi_depth"] / df["koi_srad"]
    if {"koi_prad", "koi_srad"}.issubset(df.columns):
        df["prad_srad_ratio"] = df["koi_prad"] / df["koi_srad"]
    if {"koi_sma", "koi_srad"}.issubset(df.columns):
        df["a_scaled"] = df["koi_sma"] / df["koi_srad"]

    if {"koi_gmag", "koi_rmag"}.issubset(df.columns):
        df["g_r"] = df["koi_gmag"] - df["koi_rmag"]
    if {"koi_rmag", "koi_imag"}.issubset(df.columns):
        df["r_i"] = df["koi_rmag"] - df["koi_imag"]
    if {"koi_jmag", "koi_hmag"}.issubset(df.columns):
        df["j_h"] = df["koi_jmag"] - df["koi_hmag"]

    if {"koi_duration", "koi_period"}.issubset(df.columns):
        df["dur_period_ratio"] = df["koi_duration"] / df["koi_period"]

    return df


def add_habitability(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    need = {"koi_srad", "koi_steff", "koi_sma"}
    if need.issubset(df.columns):
        L = (df["koi_srad"] ** 2) * (df["koi_steff"] / 5777.0) ** 4
        d_center = np.sqrt(L)
        df["stellar_lum"] = L
        df["d_center"] = d_center
        df["d_inner"] = 0.95 * d_center
        df["d_outer"] = 1.37 * d_center
        df["habitable_zone"] = ((df["koi_sma"] >= df["d_inner"]) & (df["koi_sma"] <= df["d_outer"])).astype(int)
    else:
        df["stellar_lum"] = np.nan
        df["d_center"] = np.nan
        df["d_inner"] = np.nan
        df["d_outer"] = np.nan
        df["habitable_zone"] = 0
    return df

# ------------------------------- Utilities -------------------------------------

def threshold_search(y_true, y_proba, metric="f1"):
    ps, rs, th = precision_recall_curve(y_true, y_proba)
    f1s = 2 * ps[:-1] * rs[:-1] / (ps[:-1] + rs[:-1] + 1e-12)
    if metric == "f1":
        i = np.nanargmax(f1s)
        return float(th[i]), float(f1s[i]), float(ps[i]), float(rs[i])
    i = np.nanargmax(f1s)
    return float(th[i]), float(f1s[i]), float(ps[i]), float(rs[i])

def try_import_smote():
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore
        return SMOTE
    except Exception:
        return None

def extract_name_series(df: pd.DataFrame) -> pd.Series:
    candidates = ["kepler_name", "pl_name", "kepoi_name", "koi_name", "tic_id"]
    for c in candidates:
        if c in df.columns:
            s = df[c].astype(str).fillna("unknown")
            s = s.replace({"": "unknown", "nan": "unknown"})
            return s
    return pd.Series([f"obj_{i}" for i in range(len(df))], index=df.index)

# ------------------------------- Classifier ------------------------------------

class ExoplanetClassifier:
    def __init__(self, contamination=0.08, rf_trees=800, fast=False):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
        self.fast = fast

        self.anomaly_detector = IsolationForest(
            n_estimators=100 if fast else 200,
            max_samples=0.5 if fast else "auto",
            contamination=contamination, n_jobs=-1, random_state=None
        )

        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=4000, class_weight="balanced", n_jobs=-1),
            "DecisionTree": DecisionTreeClassifier(random_state=None, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(
                n_estimators=rf_trees, random_state=None, n_jobs=-1, class_weight="balanced_subsample",
            ),
            "HistGB": HistGradientBoostingClassifier(random_state=None)
        }

        if fast:
            self.models.pop("DecisionTree", None)
            self.models["RandomForest"].set_params(max_depth=20, max_features="sqrt")

        self.best_name = None
        self.best_estimator = None
        self.best_threshold = 0.5

    # --------------------------- Data handling ---------------------------

    def load(self, path):
        print(f"Loading data from {path} ...")
        df = pd.read_csv(path)
        print(f"Data: {df.shape[0]} rows, {df.shape[1]} cols")
        return df

    def targetize(self, df, target_col="koi_disposition"):
        if target_col not in df.columns:
            raise ValueError(f"missing target {target_col}")
        df = df.copy()
        df["target"] = (df[target_col] == "CONFIRMED").astype(int)
        df["_name"] = extract_name_series(df)
        print("Target counts:\n", df["target"].value_counts())
        print(f"Positive rate: {df['target'].mean():.2%}")
        return df

    def safe_columns(self, df):
        base = [
            "koi_period","koi_duration","koi_depth","koi_ror",
            "koi_srho","koi_prad","koi_sma","koi_incl",
            "koi_teq","koi_insol","koi_dor",
            "koi_steff","koi_slogg","koi_smet",
            "koi_srad","koi_smass",
            "ra","dec",
            "koi_kepmag","koi_gmag","koi_rmag","koi_imag","koi_zmag",
            "koi_jmag","koi_hmag","koi_kmag",
            "log_koi_period","log_koi_depth","log_koi_prad","log_koi_insol",
            "depth_norm","prad_srad_ratio","a_scaled",
            "g_r","r_i","j_h","dur_period_ratio",
            "stellar_lum","d_center","d_inner","d_outer","habitable_zone",
        ]
        keep = [c for c in base if c in df.columns]
        return df[keep + ["target"]]

    def preprocess(self, df, fit=True):
        names = df["_name"] if "_name" in df.columns else extract_name_series(df)

        df = add_engineered_features(df)
        df = add_habitability(df)
        df = self.safe_columns(df)

        y = df["target"].values
        X = df.drop(columns=["target"])
        hz = X["habitable_zone"].fillna(0).astype(int) if "habitable_zone" in X.columns else pd.Series(np.zeros(len(X), dtype=int))

        X = X.dropna(axis=1, how="all")

        if fit:
            self.categorical_features = X.select_dtypes(include=["object","category"]).columns.tolist()
            self.numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
            print(f"Numerical={len(self.numerical_features)} Categorical={len(self.categorical_features)}")

        Z = X.copy()
        for col in self.categorical_features:
            if fit:
                le = LabelEncoder()
                Z[col] = le.fit_transform(Z[col].fillna("missing"))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                Z[col] = Z[col].fillna("missing").apply(lambda v: le.transform([v])[0] if v in le.classes_ else -1)

        if fit:
            Z[self.numerical_features] = self.imputer.fit_transform(Z[self.numerical_features])
        else:
            Z[self.numerical_features] = self.imputer.transform(Z[self.numerical_features])

        if fit:
            self.feature_names = Z.columns.tolist()

        Z = self.scaler.fit_transform(Z) if fit else self.scaler.transform(Z)
        Z = Z.astype(np.float32)

        names = names.reset_index(drop=True)
        hz = hz.reset_index(drop=True)
        return Z, y, hz, names

    # --------------------------- Monte Carlo CV --------------------------

    def mc_cv(self, X, y, iters=60, test_size=0.2):
        print(f"\n{'='*80}\nMonte Carlo CV: {iters} iters, test_size={test_size}\n{'='*80}")
        scores = {k: {"ap":[], "f1":[]} for k in self.models}
        splits = [train_test_split(X, y, test_size=test_size, stratify=y, random_state=None) for _ in range(iters)]

        for Xtr, Xte, ytr, yte in splits:
            for name, base in self.models.items():
                m = base.__class__(**base.get_params())
                m.fit(Xtr, ytr)
                p = m.predict_proba(Xte)[:,1] if hasattr(m, "predict_proba") else m.predict(Xte)
                ap = average_precision_score(yte, p)
                thr, f1, _, _ = threshold_search(yte, p, metric="f1")
                scores[name]["ap"].append(ap)
                scores[name]["f1"].append(f1)

        best, best_name = -1, None
        print(f"\n{'='*80}\nMC-CV summary (mean ± std)\n{'='*80}")
        for name, d in scores.items():
            ap_mu, ap_sd = np.mean(d["ap"]), np.std(d["ap"])
            f1_mu, f1_sd = np.mean(d["f1"]), np.std(d["f1"])
            print(f"{name:18s} AP={ap_mu:.4f}±{ap_sd:.4f}  F1={f1_mu:.4f}±{f1_sd:.4f}")
            if f1_mu > best:
                best, best_name = f1_mu, name
        print(f"\nBest (MC-CV): {best_name}  mean F1={best:.4f}")
        return best_name

    # ------------------------ Hyperparameter tuning ----------------------

    def tune(self, X, y, n_iter=200, cv_folds=5):
        if self.fast:
            n_iter = min(n_iter, 60)
            cv_folds = 3
        print(f"\n{'='*80}\nRandomizedSearchCV tuning: trials={n_iter}, folds={cv_folds}\n{'='*80}")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=None)

        spaces = {
            "RandomForest": {
                "n_estimators": randint(200 if self.fast else 400, 600 if self.fast else 2000),
                "max_depth": randint(8, 24 if self.fast else 40),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt","log2",None],
                "bootstrap": [True, False],
                "class_weight": ["balanced","balanced_subsample"]
            },
            "HistGB": {
                "max_depth": randint(3, 12 if self.fast else 16),
                "learning_rate": loguniform(1e-2, 2e-1),
                "l2_regularization": loguniform(1e-4, 1.0),
                "max_iter": randint(200, 500 if self.fast else 1200)
            },
            "LogisticRegression": {
                "C": loguniform(1e-3, 1e3),
                "solver": ["lbfgs","liblinear","saga"],
                "class_weight": [None,"balanced"],
                "max_iter": [2000,4000,6000]
            },
            "DecisionTree": {
                "max_depth": randint(3, 40 if self.fast else 60),
                "min_samples_split": randint(2, 30),
                "min_samples_leaf": randint(1, 15),
                "criterion": ["gini","entropy","log_loss"],
                "class_weight": [None,"balanced"]
            }
        }

        best_name, best_score, best_est = None, -1, None
        for name, base in self.models.items():
            print(f"\nTuning {name} ...")
            rs = RandomizedSearchCV(
                estimator=base,
                param_distributions=spaces[name],
                n_iter=n_iter,
                scoring="average_precision",
                cv=cv, n_jobs=-1, verbose=1, refit=True, random_state=None
            )
            rs.fit(X, y)
            print(f"{name} best AP={rs.best_score_:.4f} params={rs.best_params_}")
            if rs.best_score_ > best_score:
                best_name, best_score, best_est = name, rs.best_score_, rs.best_estimator_

        self.best_name = best_name
        self.best_estimator = best_est
        print(f"\nBest tuned: {best_name}  AP={best_score:.4f}")

    # --------------------------- Train / Evaluate ------------------------

    def fit_final(self, Xtr, ytr, use_calibration=False):
        if use_calibration:
            self.best_estimator = CalibratedClassifierCV(self.best_estimator, method="isotonic", cv=3)
        self.best_estimator.fit(Xtr, ytr)

    def train(self, X, y, use_tuning=True, mc_iters=60, use_smote=False, calibrate=False, ensemble=False):
        print(f"\n{'='*80}\nIsolationForest filtering\n{'='*80}")
        self.anomaly_detector.fit(X)
        mask = self.anomaly_detector.predict(X) == 1
        Xf, yf = X[mask], y[mask]
        print(f"After filtering: {Xf.shape[0]} / {X.shape[0]} samples")

        if use_smote:
            SMOTE = try_import_smote()
            if SMOTE is not None:
                print("Applying SMOTE oversampling...")
                sm = SMOTE(random_state=None)
                Xf, yf = sm.fit_resample(Xf, yf)
                print("Post-SMOTE:", Xf.shape, "Pos rate:", yf.mean())
            else:
                print("SMOTE not available (install imbalanced-learn). Skipping.")

        if use_tuning:
            self.tune(Xf, yf, n_iter=200, cv_folds=5)
        else:
            self.best_name = self.mc_cv(Xf, yf, iters=mc_iters)
            self.best_estimator = self.models[self.best_name]

        Xtr, Xva, ytr, yva = train_test_split(Xf, yf, test_size=0.2, stratify=yf, random_state=None)
        self.fit_final(Xtr, ytr, use_calibration=calibrate and not self.fast)

        proba_va = self.proba(Xva)
        thr, f1, p, r = threshold_search(yva, proba_va, metric="f1")
        self.best_threshold = max(0.05, min(0.95, thr))
        print(f"Tuned threshold: {self.best_threshold:.3f}  (val F1={f1:.4f}, P={p:.4f}, R={r:.4f})")

        if ensemble and not self.fast:
            print("\nBuilding soft-voting ensemble...")
            m_rf = self.models["RandomForest"].__class__(**self.models["RandomForest"].get_params())
            m_lr = self.models["LogisticRegression"].__class__(**self.models["LogisticRegression"].get_params())
            m_hg = self.models["HistGB"].__class__(**self.models["HistGB"].get_params())
            ens = VotingClassifier(estimators=[("rf", m_rf),("lr", m_lr),("hgb", m_hg)], voting="soft", n_jobs=-1)
            ens.fit(Xf, yf)
            p_ens = ens.predict_proba(Xva)[:,1]
            ap_ens = average_precision_score(yva, p_ens)
            ap_best = average_precision_score(yva, proba_va)
            if ap_ens >= ap_best:
                self.best_estimator = ens
                proba_va = p_ens
                thr, f1, p, r = threshold_search(yva, proba_va, metric="f1")
                self.best_threshold = max(0.05, min(0.95, thr))
                self.best_name = "SoftVotingEnsemble"
                print(f"Ensemble selected. Val AP={ap_ens:.4f}. New threshold={self.best_threshold:.3f}")
            else:
                print(f"Kept tuned single model. Val AP={ap_best:.4f}")

    def proba(self, X):
        if hasattr(self.best_estimator, "predict_proba"):
            return self.best_estimator.predict_proba(X)[:,1]
        try:
            return self.best_estimator.decision_function(X)
        except Exception:
            return self.best_estimator.predict(X).astype(float)

    def evaluate(self, X, y, hz_flag_test: pd.Series, names_test: pd.Series):
        print(f"\n{'='*80}\nFinal Evaluation\n{'='*80}")
        mask = self.anomaly_detector.predict(X) == 1
        Xf, yf = X[mask], y[mask]
        idx = mask.nonzero()[0]
        hz_f = hz_flag_test.iloc[idx].reset_index(drop=True)
        names_f = names_test.iloc[idx].reset_index(drop=True)

        proba = self.proba(Xf)
        preds = (proba >= self.best_threshold).astype(int)

        ap = average_precision_score(yf, proba)
        print(f"Average Precision (PR-AUC): {ap:.4f}")
        print(classification_report(yf, preds, target_names=["Not Confirmed","Confirmed"]))
        print("Confusion Matrix:\n", confusion_matrix(yf, preds))

        # HZ-only names
        true_idx = np.where(yf == 1)[0]
        pred_idx = np.where(preds == 1)[0]
        true_hz_idx = true_idx[np.where(hz_f.iloc[true_idx].values == 1)[0]]
        pred_hz_idx = pred_idx[np.where(hz_f.iloc[pred_idx].values == 1)[0]]

        true_hz_names = names_f.iloc[true_hz_idx].tolist()
        pred_hz_names = names_f.iloc[pred_hz_idx].tolist()

        n_true = int(len(true_idx))
        n_pred = int(len(pred_idx))
        n_true_hz = int(len(true_hz_idx))
        n_pred_hz = int(len(pred_hz_idx))

        print("\n" + "-"*80)
        print("Habitable Zone Summary (test, after anomaly filtering)")
        print("-"*80)
        pr_true = (n_true_hz / n_true * 100) if n_true else 0.0
        pr_pred = (n_pred_hz / n_pred * 100) if n_pred else 0.0
        print(f"True CONFIRMED planets in HZ: {n_true_hz}  ({pr_true:.2f}%)")
        print(f"  HZ names (true): {true_hz_names}")
        print(f"Predicted planets in HZ:      {n_pred_hz}  ({pr_pred:.2f}%)")
        print(f"  HZ names (pred): {pred_hz_names}")

        return {
            "ap": float(ap),
            "true_planets": n_true, "true_hz_planets": n_true_hz,
            "pred_planets": n_pred, "pred_hz_planets": n_pred_hz,
            "true_hz_planet_names": true_hz_names,
            "pred_hz_planet_names": pred_hz_names
        }

    def save(self, outdir="models"):
        os.makedirs(outdir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(outdir, "scaler.pkl"))
        joblib.dump(self.imputer, os.path.join(outdir, "imputer.pkl"))
        joblib.dump(self.label_encoders, os.path.join(outdir, "label_encoders.pkl"))
        joblib.dump(self.anomaly_detector, os.path.join(outdir, "anomaly.pkl"))
        joblib.dump(self.best_estimator, os.path.join(outdir, "model.pkl"))
        joblib.dump({"name": self.best_name, "threshold": self.best_threshold}, os.path.join(outdir, "model_meta.pkl"))
        print("Models saved to", outdir)

# ---------------------------------- Main ---------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cumulative_2025.10.03_09.12.20.csv")
    ap.add_argument("--models", default="models")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--contamination", type=float, default=0.08)
    ap.add_argument("--rf-trees", type=int, default=800)
    ap.add_argument("--mc-iters", type=int, default=60)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--smote", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--fast", action="store_true", help="Run with reduced iterations and lighter models for speed")
    return ap.parse_args()

def main():
    args = parse_args()

    print("="*80)
    print("Exoplanet Classification: Astro Features + HZ + MC-CV + Tuning + Thresholding")
    print("="*80)

    if not os.path.exists(args.data):
        print(f"Error: data file not found: {args.data}")
        return

    if args.fast:
        print("\nFAST MODE ENABLED: reduced iterations, lighter trees, no ensemble or calibration.\n")
        args.mc_iters = 8
        args.rf_trees = min(args.rf_trees, 300)
        args.tune = False if not args.tune else True
        args.smote = False
        args.calibrate = False
        args.ensemble = False
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

    clf = ExoplanetClassifier(contamination=args.contamination, rf_trees=args.rf_trees, fast=args.fast)

    df = clf.load(args.data)
    df = clf.targetize(df)

    df_hz = add_habitability(df)
    total_true = int((df_hz["koi_disposition"] == "CONFIRMED").sum())
    total_true_hz = int(((df_hz["koi_disposition"] == "CONFIRMED") & (df_hz["habitable_zone"] == 1)).sum())
    pct = (total_true_hz / total_true * 100) if total_true else 0.0
    print("\nGlobal HZ among true CONFIRMED (full data):")
    print(f"  True planets: {total_true} | In HZ: {total_true_hz} ({pct:.2f}%)")

    X, y, hz, names = clf.preprocess(df, fit=True)

    Xtr, Xte, ytr, yte, hz_tr, hz_te, names_tr, names_te = train_test_split(
        X, y, hz, names, test_size=args.test_size, stratify=y, random_state=None
    )
    print(f"\nTrain {Xtr.shape[0]} | Test {Xte.shape[0]}")

    clf.train(Xtr, ytr, use_tuning=args.tune, mc_iters=args.mc_iters,
              use_smote=args.smote, calibrate=args.calibrate, ensemble=args.ensemble)

    stats = clf.evaluate(Xte, yte, hz_te, names_te)

    clf.save(args.models)
    print("="*80, "\nTraining complete!")
    print("\nSummary:", stats)

if __name__ == "__main__":
    main()
