"""
Exoplanet classification with:
- Safe astrophysical features only
- Feature engineering + Habitable Zone (HZ) flag
- IsolationForest anomaly filter (protects HZ positives) or --no-anomaly
- Optional SMOTE oversampling for imbalance
- Monte Carlo CV with PR-AUC + F1 (parallel via joblib)
- Hyperparameter tuning (RandomizedSearchCV)
- Optional probability calibration (isotonic)
- Threshold tuning: F1 by default; --recall-first uses F2 + dual thresholds (HZ vs non-HZ)
- Optional soft-voting ensemble
- Prints ONLY planet names that are in the HZ
- FAST mode to reduce runtime
- Optional fast classifiers: LightGBM or XGBoost

Examples:
  python train.py --fast
  python train.py --recall-first --no-anomaly
  python train.py --lightgbm --tune --mc-iters 24
"""

import os
import argparse
import warnings
import json
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import IsolationForest, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import randint, loguniform
from joblib import Parallel, delayed
import joblib

warnings.filterwarnings("ignore")

# ----------------------------- Optional libs -----------------------------------
try:
    import lightgbm as lgb  # type: ignore
    _HAVE_LGB = True
except Exception:
    _HAVE_LGB = False

try:
    import xgboost as xgb  # type: ignore
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False


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

def threshold_search(y_true, y_proba, beta=1.0, target_recall=None):
    """Return threshold, best_score, precision, recall. If target_recall set, pick last threshold hitting it."""
    p, r, th = precision_recall_curve(y_true, y_proba)
    if target_recall is not None:
        idx = np.where(r >= target_recall)[0]
        if len(idx):
            j = idx[-1]
            best = (1 + beta**2) * p[j] * r[j] / (beta**2 * p[j] + r[j] + 1e-12)
            return float(th[j-1] if j > 0 else 0.0), float(best), float(p[j]), float(r[j])
    fbeta = (1 + beta**2) * p[:-1] * r[:-1] / (beta**2 * p[:-1] + r[:-1] + 1e-12)
    j = np.nanargmax(fbeta)
    return float(th[j]), float(fbeta[j]), float(p[j]), float(r[j])

def try_import_smote():
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore
        return SMOTE
    except Exception:
        return None

def extract_name_series(df: pd.DataFrame) -> pd.Series:
    """Robust name extraction to avoid 'unknown' in outputs."""
    def clean_str(col):
        return (
            df[col].astype(str).str.strip()
              .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan})
        )

    s = pd.Series(index=df.index, dtype=object)

    # Preferred explicit names
    for col in ["kepler_name", "pl_name", "koi_name", "kepoi_name"]:
        if col in df.columns:
            c = clean_str(col)
            s = s.where(s.notna(), c)

    # Build names from numeric mission IDs
    def build_from_num(col, prefix):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            return v.apply(lambda x: f"{prefix} {int(x)}" if pd.notna(x) else np.nan)
        return pd.Series(np.nan, index=df.index)

    for col, pref in [
        ("kepid", "KIC"), ("kic_kepler_id", "KIC"), ("kic", "KIC"),
        ("tic_id", "TIC"), ("tic", "TIC"),
        ("epic_id", "EPIC"), ("epic", "EPIC"),
    ]:
        cand = build_from_num(col, pref)
        s = s.where(s.notna(), cand)

    # Final deterministic fallback
    fallback = pd.Series([f"obj_{i}" for i in range(len(df))], index=df.index)
    s = s.where(s.notna(), fallback)
    return s

# ------------------------------- Classifier ------------------------------------

class ExoplanetClassifier:
    def __init__(self, contamination=0.08, rf_trees=800, fast=False, use_lgb=False, use_xgb=False, recall_first=False, no_anomaly=False):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
        self.fast = fast
        self.use_lgb = use_lgb and _HAVE_LGB
        self.use_xgb = use_xgb and _HAVE_XGB
        self.recall_first = recall_first
        self.no_anomaly = no_anomaly

        self.anomaly_detector = IsolationForest(
            n_estimators=100 if fast else 200,
            max_samples=0.5 if fast else "auto",
            contamination=contamination, n_jobs=-1, random_state=None
        )

        # Base models
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=4000, class_weight="balanced", n_jobs=-1),
            "DecisionTree": DecisionTreeClassifier(random_state=None, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(
                n_estimators=rf_trees, random_state=None, n_jobs=-1, class_weight="balanced_subsample",
            ),
            "HistGB": HistGradientBoostingClassifier(random_state=None)
        }

        # Faster boosters
        if self.use_lgb:
            self.models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=600 if not fast else 300,
                objective="binary",
                learning_rate=0.05 if not fast else 0.08,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1
            )
        if self.use_xgb:
            self.models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=600 if not fast else 300,
                tree_method="hist",
                objective="binary:logistic",
                learning_rate=0.05 if not fast else 0.08,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss",
                n_jobs=-1
            )

        if fast:
            self.models.pop("DecisionTree", None)
            self.models["RandomForest"].set_params(max_depth=20, max_features="sqrt")

        self.best_name = None
        self.best_estimator = None
        self.best_threshold = 0.5
        self.best_threshold_hz = 0.5  # dual threshold for HZ rows
        self._hz_train = None  # filled after split

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

    # --------------------------- Monte Carlo CV (parallel) ----------------------

    @staticmethod
    def _eval_split(models_specs, Xtr, Xte, ytr, yte):
        out = {}
        for name, (cls, params) in models_specs.items():
            m = cls(**params)
            m.fit(Xtr, ytr)
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(Xte)[:, 1]
            else:
                p = m.predict(Xte)
            ap = average_precision_score(yte, p)
            _, f1, _, _ = threshold_search(yte, p, beta=1.0)
            out[name] = (ap, f1)
        return out

    def mc_cv(self, X, y, iters=60, test_size=0.2, n_jobs=-1, verbose=0):
        print(f"\n{'='*80}\nMonte Carlo CV (parallel): {iters} iters, test_size={test_size}\n{'='*80}")
        splits = [train_test_split(X, y, test_size=test_size, stratify=y, random_state=None) for _ in range(iters)]
        models_specs = {name: (est.__class__, est.get_params()) for name, est in self.models.items()}

        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self._eval_split)(models_specs, Xtr, Xte, ytr, yte) for (Xtr, Xte, ytr, yte) in splits
        )

        # FIX: build scores dict without using undefined 'name'
        scores = {model_name: {"ap": [], "f1": []} for model_name in self.models}
        for res in results:
            for model_name, (ap, f1) in res.items():
                scores[model_name]["ap"].append(ap)
                scores[model_name]["f1"].append(f1)

        best, best_name = -1, None
        print(f"\n{'='*80}\nMC-CV summary (mean ± std)\n{'='*80}")
        for model_name, d in scores.items():
            ap_mu, ap_sd = float(np.mean(d["ap"])), float(np.std(d["ap"]))
            f1_mu, f1_sd = float(np.mean(d["f1"])), float(np.std(d["f1"]))
            print(f"{model_name:18s} AP={ap_mu:.4f}±{ap_sd:.4f}  F1={f1_mu:.4f}±{f1_sd:.4f}")
            if f1_mu > best:
                best, best_name = f1_mu, model_name
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

        if self.use_lgb:
            spaces["LightGBM"] = {
                "n_estimators": randint(200, 1200),
                "num_leaves": randint(16, 256),
                "max_depth": randint(-1, 24),
                "learning_rate": loguniform(1e-2, 2e-1),
                "min_child_samples": randint(5, 50),
                "subsample": loguniform(0.5, 1.0),
                "colsample_bytree": loguniform(0.5, 1.0),
                "reg_alpha": loguniform(1e-4, 1.0),
                "reg_lambda": loguniform(1e-4, 1.0),
            }
        if self.use_xgb:
            spaces["XGBoost"] = {
                "n_estimators": randint(200, 1200),
                "max_depth": randint(3, 16),
                "learning_rate": loguniform(1e-2, 2e-1),
                "min_child_weight": randint(1, 12),
                "subsample": loguniform(0.5, 1.0),
                "colsample_bytree": loguniform(0.5, 1.0),
                "reg_alpha": loguniform(1e-4, 1.0),
                "reg_lambda": loguniform(1e-4, 1.0),
            }

        best_name, best_score, best_est = None, -1, None
        for name, base in self.models.items():
            if name not in spaces:
                print(f"\nSkipping tuning for {name} (no search space).")
                continue
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

        if best_est is None:
            best_name = self.mc_cv(X, y, iters=12 if self.fast else 30)
            best_est = self.models[best_name]

        self.best_name = best_name
        self.best_estimator = best_est
        print(f"\nBest tuned: {best_name}")

    # --------------------------- Train / Evaluate ------------------------

    def tune_dual_threshold(self, y_va, p_va, hz_va, base_thr):
        """Lower threshold for HZ rows to include all true HZ positives on validation."""
        thr_nonhz = base_thr
        thr_hz = base_thr
        pos_hz = (y_va == 1) & (hz_va == 1)
        if np.any(pos_hz):
            min_pos = np.min(p_va[pos_hz])
            thr_hz = max(0.01, float(min_pos) - 1e-6)
        return float(thr_hz), float(thr_nonhz)

    def fit_final(self, Xtr, ytr, use_calibration=False, hz_tr=None):
        est = self.best_estimator
        if use_calibration:
            est = CalibratedClassifierCV(est, method="isotonic", cv=3)
            self.best_estimator = est

        hz_arr = None if hz_tr is None else np.asarray(hz_tr)
        fit_kwargs = {}
        if hz_arr is not None:
            w = np.ones_like(ytr, dtype=float)
            w[(ytr == 1) & (hz_arr == 1)] *= 3.0
            if "sample_weight" in est.fit.__code__.co_varnames:
                fit_kwargs["sample_weight"] = w

        est.fit(Xtr, ytr, **fit_kwargs)

    def train(self, X, y, hz, use_tuning=True, mc_iters=60, use_smote=False, calibrate=False, ensemble=False):
        print(f"\n{'='*80}\nAnomaly filtering\n{'='*80}")

        if self.no_anomaly:
            mask_if = np.ones(len(X), dtype=bool)
            print("IsolationForest disabled (--no-anomaly).")
        else:
            self.anomaly_detector.fit(X)
            mask_if = (self.anomaly_detector.predict(X) == 1)

        if self._hz_train is not None:
            protect = ((y == 1) & (np.asarray(self._hz_train) == 1))
            mask_if = np.logical_or(mask_if, protect)
            print(f"Protected HZ positives in train: {int(protect.sum())}")

        hz = np.asarray(hz, dtype=int)

        Xf, yf, hzf = X[mask_if], y[mask_if], hz[mask_if]
        print(f"After filtering: {Xf.shape[0]} / {X.shape[0]} samples")

        if use_smote:
            SMOTE = try_import_smote()
            if SMOTE is not None:
                print("Applying SMOTE oversampling...")
                sm = SMOTE(random_state=None)
                Xf, yf = sm.fit_resample(Xf, yf)
                print("Post-SMOTE:", Xf.shape, "Pos rate:", float(yf.mean()))
            else:
                print("SMOTE not available. Skipping.")

        if use_tuning:
            self.tune(Xf, yf, n_iter=200, cv_folds=5)
        else:
            self.best_name = self.mc_cv(Xf, yf, iters=mc_iters, n_jobs=-1)
            self.best_estimator = self.models[self.best_name]

        Xtr, Xva, ytr, yva, hz_tr2, hz_va = train_test_split(
            Xf, yf, hzf, test_size=0.2, stratify=yf, random_state=42
        )
        self.fit_final(Xtr, ytr, use_calibration=calibrate and not self.fast, hz_tr=hz_tr2)

        proba_va = self.proba(Xva)

        if self.recall_first:
            thr, _, _, _ = threshold_search(yva, proba_va, beta=2.0)
            thr_hz, thr_nonhz = self.tune_dual_threshold(yva, proba_va, np.asarray(hz_va), base_thr=thr)
            self.best_threshold_hz = np.clip(thr_hz, 0.01, 0.95)
            self.best_threshold = np.clip(thr_nonhz, 0.01, 0.95)
            print(f"Tuned thresholds (recall-first): HZ={self.best_threshold_hz:.3f}  non-HZ={self.best_threshold:.3f}")
        else:
            thr, best, p, r = threshold_search(yva, proba_va, beta=1.0)
            self.best_threshold = np.clip(thr, 0.01, 0.95)
            self.best_threshold_hz = self.best_threshold
            print(f"Tuned threshold (F1): {self.best_threshold:.3f}  (val score={best:.4f}, P={p:.4f}, R={r:.4f})")

        if ensemble and not self.fast:
            print("\nBuilding soft-voting ensemble...")
            estims = []
            if "RandomForest" in self.models:
                estims.append(("rf", self.models["RandomForest"].__class__(**self.models["RandomForest"].get_params())))
            if "LightGBM" in self.models:
                estims.append(("lgb", self.models["LightGBM"].__class__(**self.models["LightGBM"].get_params())))
            if "XGBoost" in self.models:
                estims.append(("xgb", self.models["XGBoost"].__class__(**self.models["XGBoost"].get_params())))
            if "HistGB" in self.models and not estims:
                estims.append(("hgb", self.models["HistGB"].__class__(**self.models["HistGB"].get_params())))
            if len(estims) >= 2:
                ens = VotingClassifier(estimators=estims, voting="soft", n_jobs=-1)
                ens.fit(Xf, yf)
                p_ens = ens.predict_proba(Xva)[:, 1]
                ap_ens = average_precision_score(yva, p_ens)
                ap_best = average_precision_score(yva, proba_va)
                if ap_ens >= ap_best:
                    self.best_estimator = ens
                    if self.recall_first:
                        thr_hz, thr_nonhz = self.tune_dual_threshold(yva, p_ens, np.asarray(hz_va), base_thr=self.best_threshold)
                        self.best_threshold_hz = np.clip(thr_hz, 0.01, 0.95)
                        self.best_threshold = np.clip(thr_nonhz, 0.01, 0.95)
                    else:
                        thr, _, _, _ = threshold_search(yva, p_ens, beta=1.0)
                        self.best_threshold = np.clip(thr, 0.01, 0.95)
                        self.best_threshold_hz = self.best_threshold
                    print(f"Ensemble selected. Thresholds: HZ={self.best_threshold_hz:.3f} non-HZ={self.best_threshold:.3f}")
                else:
                    print("Kept tuned single model.")

    def proba(self, X):
        if hasattr(self.best_estimator, "predict_proba"):
            return self.best_estimator.predict_proba(X)[:,1]
        try:
            return self.best_estimator.decision_function(X)
        except Exception:
            return self.best_estimator.predict(X).astype(float)

    def predict_labels(self, X, hz_flag):
        p = self.proba(X)
        thr_vec = np.where(np.asarray(hz_flag) == 1, self.best_threshold_hz, self.best_threshold)
        return (p >= thr_vec).astype(int), p

    def evaluate(self, X, y, hz_flag_test: pd.Series, names_test: pd.Series):

        print(f"{'='*80}Final Evaluation{'='*80}")

        pre_true_hz = int(((y == 1) & (np.asarray(hz_flag_test) == 1)).sum())
        print(f"[Info] TEST split BEFORE anomaly filter: true HZ positives = {pre_true_hz}")

        if self.no_anomaly:
            mask_if = np.ones(len(X), dtype=bool)
        else:
            mask_if = (self.anomaly_detector.predict(X) == 1)

        Xf, yf = X[mask_if], y[mask_if]
        idx = np.where(mask_if)[0]
        hz_f = np.asarray(hz_flag_test)[idx]
        names_f = names_test.iloc[idx].reset_index(drop=True)

        anomalies_detected = int((~mask_if).sum())
        filtered_samples = int(len(Xf))
        total_samples = int(len(X))

        print(f"[Info] After anomaly filter: kept {filtered_samples} / {total_samples} test rows")
        print(f"[Info] True HZ positives after filter: {int(((yf==1)&(hz_f==1)).sum())}")

        preds, proba = self.predict_labels(Xf, hz_f)

        ap = average_precision_score(yf, proba)
        precision = precision_score(yf, preds, zero_division=0)
        recall = recall_score(yf, preds, zero_division=0)
        f1 = f1_score(yf, preds, zero_division=0)
        accuracy = accuracy_score(yf, preds)
        cm = confusion_matrix(yf, preds, labels=[0, 1])

        print(f"Average Precision (PR-AUC): {ap:.4f}")
        print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}  Accuracy: {accuracy:.4f}")
        print(classification_report(yf, preds, target_names=["Not Confirmed","Confirmed"]))
        print("Confusion Matrix:\n", cm)

        # HZ-only names
        true_idx = np.where(yf == 1)[0]
        pred_idx = np.where(preds == 1)[0]
        true_hz_idx = true_idx[np.where(hz_f[true_idx] == 1)[0]]
        pred_hz_idx = pred_idx[np.where(hz_f[pred_idx] == 1)[0]]

        true_hz_names = names_f.iloc[true_hz_idx].tolist()
        pred_hz_names = names_f.iloc[pred_hz_idx].tolist()

        n_true = int(len(true_idx))
        n_pred = int(len(pred_idx))
        n_true_hz = int(len(true_hz_idx))
        n_pred_hz = int(len(pred_hz_idx))

        print("" + "-"*80)
        print("Habitable Zone Summary (test, consistent split)")
        print("-"*80)
        pr_true = (n_true_hz / n_true * 100) if n_true else 0.0
        pr_pred = (n_pred_hz / n_pred * 100) if n_pred else 0.0
        print(f"True CONFIRMED planets in HZ: {n_true_hz}  ({pr_true:.2f}%)")
        print(f"  HZ names (true): {true_hz_names}")
        print(f"Predicted planets in HZ:      {n_pred_hz}  ({pr_pred:.2f}%)")
        print(f"  HZ names (pred): {pred_hz_names}")

        model_label = self.best_name or (self.best_estimator.__class__.__name__ if self.best_estimator is not None else "unknown")
        tn, fp, fn, tp = cm.ravel()

        predictions = []
        order = np.argsort(proba)[::-1]
        for idx in order[: min(25, len(order))]:
            name = names_f.iloc[idx]
            if pd.isna(name):
                name = f"Object-{idx}"
            predictions.append({
                "name": str(name),
                "prediction": "CONFIRMED" if int(preds[idx]) == 1 else "NOT_CONFIRMED",
                "actual": "CONFIRMED" if int(yf[idx]) == 1 else "NOT_CONFIRMED",
                "confidence": float(proba[idx]),
                "hz": bool(int(hz_f[idx]) == 1),
                "anomaly": False
            })

        summary = {
            "average_precision": float(ap),
            "total_samples": total_samples,
            "filtered_samples": filtered_samples,
            "confirmed_predictions": int(np.sum(preds)),
            "hz_candidates": n_pred_hz,
            "anomalies_detected": anomalies_detected,
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(accuracy),
            "positive_rate": float(np.mean(y)),
        }

        model_overview = {
            "name": model_label,
            "metrics": {
                "average_precision": float(ap),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "accuracy": float(accuracy),
                "threshold": float(self.best_threshold),
                "threshold_hz": float(self.best_threshold_hz),
            },
        }

        performance = {
            "models": [model_label],
            "ap_scores": [float(ap)],
            "f1_scores": [float(f1)],
            "precision_scores": [float(precision)],
            "recall_scores": [float(recall)],
            "accuracy_scores": [float(accuracy)],
            "thresholds": [float(self.best_threshold)],
            "thresholds_hz": [float(self.best_threshold_hz)],
        }

        confusion = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }

        hz_summary = {
            "true": true_hz_names,
            "predicted": pred_hz_names,
            "counts": {
                "true_planets": n_true,
                "predicted_planets": n_pred,
                "true_hz_planets": n_true_hz,
                "predicted_hz_planets": n_pred_hz,
            },
        }

        return {
            "summary": summary,
            "model_overview": model_overview,
            "performance": performance,
            "confusion_matrix": confusion,
            "hz_planets": hz_summary,
            "predictions": predictions,
        }

    def save(self, outdir="models"):
        os.makedirs(outdir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(outdir, "scaler.pkl"))
        joblib.dump(self.imputer, os.path.join(outdir, "imputer.pkl"))
        joblib.dump(self.label_encoders, os.path.join(outdir, "label_encoders.pkl"))
        joblib.dump(self.anomaly_detector, os.path.join(outdir, "anomaly.pkl"))
        joblib.dump(self.best_estimator, os.path.join(outdir, "model.pkl"))
        joblib.dump(
            {"name": self.best_name, "threshold": self.best_threshold, "threshold_hz": self.best_threshold_hz,
             "recall_first": self.recall_first, "no_anomaly": self.no_anomaly},
            os.path.join(outdir, "model_meta.pkl")
        )
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
    ap.add_argument("--fast", action="store_true", help="Reduced iterations and lighter models")
    ap.add_argument("--lightgbm", action="store_true", help="Include LightGBM if installed")
    ap.add_argument("--xgboost", action="store_true", help="Include XGBoost if installed")
    ap.add_argument("--recall-first", action="store_true", help="Use F2 and dual thresholds to favor HZ recall")
    ap.add_argument("--no-anomaly", action="store_true", help="Disable IsolationForest filtering")
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
        print("\nFAST MODE ENABLED.\n")
        args.mc_iters = min(args.mc_iters, 12)
        args.rf_trees = min(args.rf_trees, 300)
        args.smote = False
        args.calibrate = False
        args.ensemble = False
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

    clf = ExoplanetClassifier(
        contamination=args.contamination,
        rf_trees=args.rf_trees,
        fast=args.fast,
        use_lgb=args.lightgbm,
        use_xgb=args.xgboost,
        recall_first=args.recall_first,
        no_anomaly=args.no_anomaly
    )

    if args.lightgbm and not _HAVE_LGB:
        print("LightGBM requested but not installed. Skipping.")
    if args.xgboost and not _HAVE_XGB:
        print("XGBoost requested but not installed. Skipping.")

    df = clf.load(args.data)
    df = clf.targetize(df)

    X, y, hz, names = clf.preprocess(df, fit=True)

    # Stratify by combined class+HZ to balance rare HZ positives across splits
    strata = (y.astype(int) * 2 + np.asarray(hz).astype(int))
    Xtr, Xte, ytr, yte, hz_tr, hz_te, names_tr, names_te = train_test_split(
        X, y, np.asarray(hz), names, test_size=args.test_size, stratify=strata, random_state=42
    )
    clf._hz_train = np.asarray(hz_tr)  # for protecting HZ positives during anomaly filtering
    print(f"\nTrain {Xtr.shape[0]} | Test {Xte.shape[0]}")
    print(f"[Info] TEST split true HZ positives (before IF): {int(((yte==1)&(np.asarray(hz_te)==1)).sum())}")

    # Train and evaluate
    clf.train(Xtr, ytr, np.asarray(hz_tr), use_tuning=args.tune, mc_iters=args.mc_iters,
              use_smote=args.smote, calibrate=args.calibrate, ensemble=args.ensemble)


    stats = clf.evaluate(Xte, yte, np.asarray(hz_te), names_te)

    clf.save(args.models)

    target_counts = df["target"].value_counts().to_dict()
    training_info = {
        "best_model": stats["model_overview"]["name"],
        "threshold": round(stats["model_overview"]["metrics"]["threshold"], 3),
        "threshold_hz": round(stats["model_overview"]["metrics"]["threshold_hz"], 3),
        "train_samples": int(Xtr.shape[0]),
        "test_samples": int(Xte.shape[0]),
        "filtered_samples": stats["summary"]["filtered_samples"],
        "target_distribution": {
            "class_0": int(target_counts.get(0, 0)),
            "class_1": int(target_counts.get(1, 0)),
        },
    }

    results_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": stats["summary"],
        "model_overview": stats["model_overview"],
        "performance": stats["performance"],
        "hz_planets": stats["hz_planets"],
        "confusion_matrix": stats["confusion_matrix"],
        "training_info": training_info,
        "predictions": stats["predictions"],
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, "templates", "results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results_payload, fh, indent=2)
    print(f"Saved evaluation report to {results_path}")

    print("="*80, "Training complete!")
    print("Summary:", stats["summary"])

if __name__ == "__main__":
    main()
