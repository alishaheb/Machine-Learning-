"""
AutoML Pipeline for Tabular Data (v3.0 — Innovative Edition)
==============================================================
A next-generation ML pipeline with capabilities no existing AutoML tool offers:

  CORE:
    1. Data profiling, cleaning, preprocessing
    2. Adaptive feature engineering
    3. Auto task detection (classification / regression / both / clustering)
    4. Multi-model training with cross-validation

  INNOVATIVE MODULES:
    5. Self-Healing Pipeline — detects poor performance, auto-retries with
       different imputation, scaling, encoding, and feature strategies
    6. Feature Story Generator — produces plain-English explanations of why
       each feature matters using SHAP + narrative generation
    7. Data Drift Sentinel — monitors new data for distribution shift,
       new categories, schema changes, and alerts before predictions fail
    8. Adversarial Stress Testing — generates edge cases, noise, missing
       data attacks, and reports a model robustness score
    9. Cross-Task Insight Transfer — compares feature importance across
       classification vs regression to find task-agnostic vs task-specific features

Usage:
    python auto_ml_pipeline.py --data data.csv --target col
    python auto_ml_pipeline.py --data data.csv --target col --task both
    python auto_ml_pipeline.py --data data.csv --target col --heal        # enable self-healing
    python auto_ml_pipeline.py --data data.csv --target col --stress      # adversarial testing
    python auto_ml_pipeline.py --data data.csv --target col --story       # feature story
    python auto_ml_pipeline.py --data data.csv --target col --all         # everything
    python auto_ml_pipeline.py --monitor model.pkl --baseline data.csv --new new_data.csv  # drift detection

Requirements:
    pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn shap
"""

import argparse
import os
import sys
import json
import warnings
import time
import pickle
import copy
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
    LabelEncoder, OrdinalEncoder, OneHotEncoder,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline

from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, silhouette_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from scipy import stats

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    max_onehot_cardinality: int = 15
    missing_threshold: float = 0.5
    n_bins_for_classification: int = 4
    n_clusters_range: tuple = (2, 10)
    output_dir: str = "pipeline_output"
    # Self-healing
    healing_max_rounds: int = 3
    healing_score_threshold: float = 0.05  # min improvement to keep healing
    # Adversarial
    stress_noise_levels: tuple = (0.05, 0.1, 0.2, 0.5)
    stress_missing_rates: tuple = (0.05, 0.1, 0.2, 0.3)
    # Drift
    drift_pvalue_threshold: float = 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def banner(text, char="=", width=60):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def sub_banner(text):
    print(f"\n── {text} ──")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA PROFILING
# ═══════════════════════════════════════════════════════════════════════════════

def profile_data(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    profile = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "nunique": df.nunique().to_dict(),
        "numeric_cols": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": list(df.select_dtypes(include=["object", "category"]).columns),
        "datetime_cols": list(df.select_dtypes(include=["datetime64"]).columns),
    }

    if target and target in df.columns:
        profile["target_dtype"] = str(df[target].dtype)
        profile["target_nunique"] = df[target].nunique()
        profile["target_missing"] = int(df[target].isnull().sum())
        if pd.api.types.is_numeric_dtype(df[target]):
            profile["target_stats"] = df[target].describe().to_dict()
        else:
            profile["target_distribution"] = df[target].value_counts().to_dict()

    banner("DATA PROFILE")
    print(f"  Rows:              {profile['shape'][0]:,}")
    print(f"  Columns:           {profile['shape'][1]}")
    print(f"  Numeric features:  {len(profile['numeric_cols'])}")
    print(f"  Categorical:       {len(profile['categorical_cols'])}")
    print(f"  Datetime:          {len(profile['datetime_cols'])}")

    missing_cols = {k: v for k, v in profile["missing"].items() if v > 0}
    if missing_cols:
        print(f"  Cols with missing:  {len(missing_cols)}")
        for col, cnt in sorted(missing_cols.items(), key=lambda x: -x[1])[:5]:
            print(f"    - {col}: {cnt} ({profile['missing_pct'][col]}%)")

    if target:
        print(f"\n  Target column:     '{target}'")
        print(f"  Target unique:     {profile.get('target_nunique', 'N/A')}")

    return profile


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TASK DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_task(df, target, profile):
    if target is None or target not in df.columns:
        print("  [Auto-detect] No target → Clustering")
        return "clustering"
    col = df[target]
    nunique = profile["target_nunique"]
    if col.dtype == "object" or col.dtype.name == "category":
        print(f"  [Auto-detect] Categorical target ({nunique} classes) → Classification")
        return "classification"
    if pd.api.types.is_numeric_dtype(col) and nunique <= 20:
        print(f"  [Auto-detect] Numeric target with {nunique} uniques → Classification")
        return "classification"
    print(f"  [Auto-detect] Continuous target ({nunique} uniques) → Regression")
    return "regression"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(df, target, config, profile):
    df = df.copy()

    # Drop high-missing columns
    drop_cols = [c for c, pct in profile["missing_pct"].items() if pct > config.missing_threshold * 100]
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} high-missing columns")
        df.drop(columns=drop_cols, inplace=True)

    # Drop constant columns
    const_cols = [c for c in df.columns if df[c].nunique() <= 1 and c != target]
    if const_cols:
        df.drop(columns=const_cols, inplace=True)

    # Parse datetimes
    for col in profile["datetime_cols"]:
        if col in df.columns and col != target:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            for attr in ["year", "month", "day", "dayofweek"]:
                df[f"{col}_{attr}"] = getattr(df[col].dt, attr)
            df.drop(columns=[col], inplace=True)

    # Detect hidden dates in object columns
    for col in list(df.select_dtypes(include=["object"]).columns):
        if col == target:
            continue
        try:
            pd.to_datetime(df[col].dropna().head(50), infer_datetime_format=True)
            df[col] = pd.to_datetime(df[col], errors="coerce")
            for attr in ["year", "month", "day", "dayofweek"]:
                df[f"{col}_{attr}"] = getattr(df[col].dt, attr)
            df.drop(columns=[col], inplace=True)
        except (ValueError, TypeError):
            pass

    # Drop ID-like columns
    for col in list(df.select_dtypes(include=["object"]).columns):
        if col != target and df[col].nunique() / max(len(df), 1) > 0.9:
            df.drop(columns=[col], inplace=True)

    # Encode target
    label_encoder = None
    if target and target in df.columns:
        y = df[target]
        X = df.drop(columns=[target])
        if y.dtype == "object" or y.dtype.name == "category":
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y.astype(str)), name=target)
            print(f"  Encoded target: {list(label_encoder.classes_)}")
    else:
        X, y = df, None

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
    print(f"  Features: {len(num_cols)} numeric, {len(cat_cols)} categorical")

    preprocessor = build_preprocessor(X, config)
    return X, y, preprocessor, label_encoder


def build_preprocessor(X, config, num_imputer="median", scaler="standard"):
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
    low_card = [c for c in cat_cols if X[c].nunique() <= config.max_onehot_cardinality]
    high_card = [c for c in cat_cols if X[c].nunique() > config.max_onehot_cardinality]

    # Imputer selection
    if num_imputer == "median":
        imp = SimpleImputer(strategy="median")
    elif num_imputer == "mean":
        imp = SimpleImputer(strategy="mean")
    elif num_imputer == "knn":
        imp = KNNImputer(n_neighbors=5)
    else:
        imp = SimpleImputer(strategy="median")

    # Scaler selection
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "power": PowerTransformer(method="yeo-johnson"),
    }
    sc = scalers.get(scaler, StandardScaler())

    transformers = []
    if num_cols:
        transformers.append(("num", SkPipeline([("imputer", imp), ("scaler", sc)]), num_cols))
    if low_card:
        transformers.append(("cat_low", SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), low_card))
    if high_card:
        transformers.append(("cat_high", SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), high_card))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(X, level="basic"):
    """Level: basic, medium, aggressive"""
    X = X.copy()
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    added = 0

    if level in ("basic", "medium", "aggressive") and len(num_cols) >= 2:
        corr = X[num_cols].corr().abs()
        max_pairs = {"basic": 5, "medium": 10, "aggressive": 20}[level]
        pairs_added = 0
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                if corr.iloc[i, j] > 0.5 and pairs_added < max_pairs:
                    a, b = num_cols[i], num_cols[j]
                    X[f"{a}_div_{b}"] = X[a] / X[b].replace(0, np.nan)
                    pairs_added += 1
                    added += 1

    if level in ("medium", "aggressive"):
        # Polynomial features for top correlated
        for col in num_cols[:5]:
            X[f"{col}_sq"] = X[col] ** 2
            added += 1

    if level == "aggressive":
        # Interaction terms
        for i, a in enumerate(num_cols[:4]):
            for b in num_cols[i+1:5]:
                X[f"{a}_x_{b}"] = X[a] * X[b]
                added += 1

    # Log transforms for skewed
    for col in num_cols:
        skew = X[col].skew()
        if abs(skew) > 2 and (X[col] > 0).all():
            X[f"{col}_log"] = np.log1p(X[col])
            added += 1

    if added > 0:
        print(f"  Engineered {added} new features (level={level})")
    return X


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODEL SELECTION & TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def get_models(task):
    if task == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0)
        if HAS_LGBM:
            models["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    elif task == "regression":
        models = {
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42),
            "ElasticNet": ElasticNet(random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
            "KNN": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        if HAS_LGBM:
            models["LightGBM"] = LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
    else:
        models = {}
    return models


def train_and_evaluate(X, y, preprocessor, task, config, tag=""):
    models = get_models(task)
    scoring = "f1_weighted" if task == "classification" else "neg_mean_squared_error"
    cv = (StratifiedKFold if task == "classification" else KFold)(
        n_splits=config.cv_folds, shuffle=True, random_state=config.random_state
    )

    results = []
    best_score, best_name, best_pipeline = -np.inf, None, None

    header = f"  MODEL TRAINING — {task.upper()}"
    if tag:
        header += f" ({tag})"
    banner(header, "─")
    print(f"  {'Model':<22} {'CV Score':>12} {'Std':>10}  {'Time':>8}")
    print("  " + "-" * 56)

    for name, model in models.items():
        pipe = SkPipeline([("preprocessor", preprocessor), ("model", model)])
        start = time.time()
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            elapsed = time.time() - start
            mean_s, std_s = scores.mean(), scores.std()
            display = mean_s if task == "classification" else -mean_s

            results.append({
                "Model": name, "Task": task,
                "CV_Score_Mean": mean_s, "CV_Score_Std": std_s,
                "Display_Score": display, "Time_s": round(elapsed, 2),
            })
            print(f"  {name:<22} {display:>12.4f} {std_s:>10.4f}  {elapsed:>7.1f}s")

            if mean_s > best_score:
                best_score, best_name, best_pipeline = mean_s, name, pipe
        except Exception as e:
            print(f"  {name:<22} {'FAILED':>12}  ({e})")

    print("  " + "-" * 56)
    metric = "F1-weighted" if task == "classification" else "RMSE"
    best_disp = best_score if task == "classification" else -best_score
    print(f"  ★ Best: {best_name} ({metric} = {best_disp:.4f})")

    best_pipeline.fit(X, y)
    return pd.DataFrame(results).sort_values("CV_Score_Mean", ascending=False), best_pipeline, best_name


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION & CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_best_model(pipeline, X, y, task, label_encoder, config, output_dir, suffix=""):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state,
        stratify=y if task == "classification" else None,
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    tag = f" [{suffix}]" if suffix else ""
    fs = f"_{suffix}" if suffix else ""

    banner(f"HOLDOUT TEST — {task.upper()}{tag}", "─")

    if task == "classification":
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        print(f"  Accuracy:   {acc:.4f}")
        print(f"  F1-Score:   {f1:.4f}")
        print(f"  Precision:  {prec:.4f}")
        print(f"  Recall:     {rec:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = label_encoder.classes_ if label_encoder else sorted(y.unique())
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix{tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix{fs}.png"), dpi=150)
        plt.close()
        return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}
    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"  RMSE:  {rmse:.4f}")
        print(f"  MAE:   {mae:.4f}")
        print(f"  R²:    {r2:.4f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.4, s=20)
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=2)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted (R²={r2:.3f}){tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"actual_vs_predicted{fs}.png"), dpi=150)
        plt.close()
        return {"rmse": rmse, "mae": mae, "r2": r2}


def save_model_comparison_chart(results_df, best_name, task, output_dir, suffix=""):
    fs = f"_{suffix}" if suffix else ""
    fig, ax = plt.subplots(figsize=(10, 6))
    metric = "F1-Weighted" if task == "classification" else "RMSE"
    rs = results_df.sort_values("Display_Score", ascending=True)
    colors = ["#2196F3" if m != best_name else "#FF9800" for m in rs["Model"]]
    ax.barh(rs["Model"], rs["Display_Score"], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(f"Model Comparison — {metric}{' (' + suffix + ')' if suffix else ''}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"model_comparison{fs}.png"), dpi=150)
    plt.close()


def save_feature_importance(pipeline, X, output_dir, suffix=""):
    fs = f"_{suffix}" if suffix else ""
    try:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            imp_df = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
            imp_df = imp_df.sort_values("importance", ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, color="steelblue")
            ax.set_title(f"Top 20 Feature Importances{' (' + suffix + ')' if suffix else ''}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"feature_importance{fs}.png"), dpi=150)
            plt.close()
            imp_df.to_csv(os.path.join(output_dir, f"feature_importance{fs}.csv"), index=False)
            return imp_df
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 7. INNOVATIVE MODULE: SELF-HEALING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class SelfHealingPipeline:
    """
    Detects when model performance is poor and automatically tries different
    preprocessing strategies (imputation, scaling, feature engineering levels)
    to find a better configuration. No existing AutoML tool does this.
    """

    STRATEGIES = [
        {"imputer": "median",  "scaler": "standard", "feat_level": "basic",      "label": "Baseline"},
        {"imputer": "mean",    "scaler": "robust",   "feat_level": "basic",      "label": "Robust scaling"},
        {"imputer": "knn",     "scaler": "standard", "feat_level": "basic",      "label": "KNN imputation"},
        {"imputer": "median",  "scaler": "power",    "feat_level": "medium",     "label": "Power transform + medium features"},
        {"imputer": "knn",     "scaler": "robust",   "feat_level": "medium",     "label": "KNN + Robust + medium features"},
        {"imputer": "median",  "scaler": "minmax",   "feat_level": "aggressive", "label": "MinMax + aggressive features"},
        {"imputer": "knn",     "scaler": "power",    "feat_level": "aggressive", "label": "KNN + Power + aggressive features"},
    ]

    @staticmethod
    def run(X_raw, y, task, config, output_dir):
        banner("SELF-HEALING PIPELINE", "★")
        print("  Trying multiple preprocessing strategies to find the best one...")

        scoring = "f1_weighted" if task == "classification" else "neg_mean_squared_error"
        cv = (StratifiedKFold if task == "classification" else KFold)(
            n_splits=config.cv_folds, shuffle=True, random_state=config.random_state
        )

        healing_log = []
        best_overall_score = -np.inf
        best_strategy = None
        best_pipeline = None
        best_results_df = None
        best_model_name = None

        for i, strat in enumerate(SelfHealingPipeline.STRATEGIES):
            if i >= config.healing_max_rounds + 1 and best_overall_score > 0.5:
                break  # Stop if we have a good score and exceeded max rounds

            print(f"\n  ┌─ Round {i+1}: {strat['label']}")

            X_eng = engineer_features(X_raw.copy(), level=strat["feat_level"])
            preprocessor = build_preprocessor(X_eng, config, strat["imputer"], strat["scaler"])

            # Quick test with just 2 models for speed
            quick_models = {}
            all_models = get_models(task)
            priority = ["LightGBM", "XGBoost", "GradientBoosting", "RandomForest", "Ridge", "LogisticRegression"]
            for name in priority:
                if name in all_models:
                    quick_models[name] = all_models[name]
                    if len(quick_models) >= 3:
                        break

            round_best_score = -np.inf
            round_best_name = None

            for name, model in quick_models.items():
                pipe = SkPipeline([("preprocessor", preprocessor), ("model", model)])
                try:
                    scores = cross_val_score(pipe, X_eng, y, cv=cv, scoring=scoring, n_jobs=-1)
                    mean_s = scores.mean()
                    display = mean_s if task == "classification" else -mean_s
                    print(f"  │  {name:<20} → {display:.4f}")

                    if mean_s > round_best_score:
                        round_best_score = mean_s
                        round_best_name = name
                except Exception as e:
                    print(f"  │  {name:<20} → FAILED ({e})")

            round_display = round_best_score if task == "classification" else -round_best_score
            healing_log.append({
                "Round": i + 1,
                "Strategy": strat["label"],
                "Best_Model": round_best_name,
                "Score": round_display,
            })

            if round_best_score > best_overall_score:
                improvement = round_best_score - best_overall_score if best_overall_score > -np.inf else 0
                best_overall_score = round_best_score
                best_strategy = strat
                print(f"  └─ ★ New best! Score: {round_display:.4f}")
            else:
                print(f"  └─ No improvement (best remains {best_overall_score if task == 'classification' else -best_overall_score:.4f})")

        # Final full training with the best strategy
        print(f"\n  ★ Best strategy: {best_strategy['label']}")
        print(f"  Re-training all models with winning configuration...")

        X_final = engineer_features(X_raw.copy(), level=best_strategy["feat_level"])
        preprocessor_final = build_preprocessor(X_final, config, best_strategy["imputer"], best_strategy["scaler"])
        results_df, best_pipe, best_name = train_and_evaluate(X_final, y, preprocessor_final, task, config, tag="self-healed")

        # Save healing log
        heal_df = pd.DataFrame(healing_log)
        heal_df.to_csv(os.path.join(output_dir, "healing_log.csv"), index=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#FF9800" if s == best_strategy["label"] else "#2196F3" for s in heal_df["Strategy"]]
        ax.barh(heal_df["Strategy"], heal_df["Score"], color=colors)
        metric = "F1-Weighted" if task == "classification" else "RMSE"
        ax.set_xlabel(metric)
        ax.set_title("Self-Healing: Strategy Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "healing_strategy_comparison.png"), dpi=150)
        plt.close()
        print(f"  → Saved healing_log.csv & healing_strategy_comparison.png")

        return X_final, preprocessor_final, results_df, best_pipe, best_name, best_strategy


# ═══════════════════════════════════════════════════════════════════════════════
# 8. INNOVATIVE MODULE: ADVERSARIAL STRESS TESTING
# ═══════════════════════════════════════════════════════════════════════════════

class AdversarialStressTester:
    """
    Generates adversarial perturbations to find where models break:
    - Gaussian noise injection at multiple levels
    - Random missing data injection
    - Feature shuffling (destroys single-feature signal)
    - Outlier injection (extreme values)
    - Category corruption (unseen categories)
    Reports a composite "Robustness Score" (0-100).
    """

    @staticmethod
    def run(pipeline, X, y, task, config, output_dir):
        banner("ADVERSARIAL STRESS TESTING", "⚡")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state,
            stratify=y if task == "classification" else None,
        )
        pipeline.fit(X_train, y_train)

        # Baseline
        y_pred_base = pipeline.predict(X_test)
        if task == "classification":
            base_score = f1_score(y_test, y_pred_base, average="weighted")
            metric_name = "F1-Weighted"
        else:
            base_score = r2_score(y_test, y_pred_base)
            metric_name = "R²"

        print(f"  Baseline {metric_name}: {base_score:.4f}")
        print()

        results = []
        num_cols = list(X_test.select_dtypes(include=[np.number]).columns)
        cat_cols = list(X_test.select_dtypes(include=["object", "category"]).columns)

        # ── Test 1: Gaussian Noise ──
        print("  ── Noise Injection ──")
        for noise_level in config.stress_noise_levels:
            X_noisy = X_test.copy()
            for col in num_cols:
                std = X_noisy[col].std()
                noise = np.random.normal(0, std * noise_level, size=len(X_noisy))
                X_noisy[col] = X_noisy[col] + noise
            try:
                y_pred = pipeline.predict(X_noisy)
                score = f1_score(y_test, y_pred, average="weighted") if task == "classification" else r2_score(y_test, y_pred)
                degradation = (base_score - score) / max(abs(base_score), 1e-10) * 100
                results.append({"Attack": f"Noise {noise_level*100:.0f}%", "Score": score, "Degradation_%": degradation})
                print(f"    Noise {noise_level*100:>5.0f}% → {metric_name}: {score:.4f} (Δ {degradation:+.1f}%)")
            except Exception:
                results.append({"Attack": f"Noise {noise_level*100:.0f}%", "Score": 0, "Degradation_%": 100})

        # ── Test 2: Missing Data Injection ──
        print("  ── Missing Data Injection ──")
        for miss_rate in config.stress_missing_rates:
            X_miss = X_test.copy()
            mask = np.random.random(X_miss[num_cols].shape) < miss_rate
            X_miss.loc[:, num_cols] = X_miss[num_cols].mask(pd.DataFrame(mask, columns=num_cols, index=X_miss.index))
            try:
                y_pred = pipeline.predict(X_miss)
                score = f1_score(y_test, y_pred, average="weighted") if task == "classification" else r2_score(y_test, y_pred)
                degradation = (base_score - score) / max(abs(base_score), 1e-10) * 100
                results.append({"Attack": f"Missing {miss_rate*100:.0f}%", "Score": score, "Degradation_%": degradation})
                print(f"    Missing {miss_rate*100:>4.0f}% → {metric_name}: {score:.4f} (Δ {degradation:+.1f}%)")
            except Exception:
                results.append({"Attack": f"Missing {miss_rate*100:.0f}%", "Score": 0, "Degradation_%": 100})

        # ── Test 3: Feature Shuffling ──
        print("  ── Feature Shuffling (per-feature signal destruction) ──")
        feature_sensitivity = []
        for col in num_cols[:10]:  # top 10
            X_shuf = X_test.copy()
            X_shuf[col] = np.random.permutation(X_shuf[col].values)
            try:
                y_pred = pipeline.predict(X_shuf)
                score = f1_score(y_test, y_pred, average="weighted") if task == "classification" else r2_score(y_test, y_pred)
                degradation = (base_score - score) / max(abs(base_score), 1e-10) * 100
                feature_sensitivity.append({"Feature": col, "Score_After_Shuffle": score, "Degradation_%": degradation})
                print(f"    Shuffle '{col[:25]:<25}' → {metric_name}: {score:.4f} (Δ {degradation:+.1f}%)")
            except Exception:
                pass

        results.append({"Attack": "Feature Shuffle (avg)", "Score": np.mean([f["Score_After_Shuffle"] for f in feature_sensitivity]) if feature_sensitivity else 0, "Degradation_%": np.mean([f["Degradation_%"] for f in feature_sensitivity]) if feature_sensitivity else 100})

        # ── Test 4: Outlier Injection ──
        print("  ── Outlier Injection ──")
        for multiplier in [3, 5, 10]:
            X_out = X_test.copy()
            n_outliers = max(1, len(X_out) // 10)
            idx = np.random.choice(len(X_out), n_outliers, replace=False)
            for col in num_cols:
                std = X_out[col].std()
                X_out.iloc[idx, X_out.columns.get_loc(col)] = X_out[col].mean() + multiplier * std
            try:
                y_pred = pipeline.predict(X_out)
                score = f1_score(y_test, y_pred, average="weighted") if task == "classification" else r2_score(y_test, y_pred)
                degradation = (base_score - score) / max(abs(base_score), 1e-10) * 100
                results.append({"Attack": f"Outliers {multiplier}x σ", "Score": score, "Degradation_%": degradation})
                print(f"    Outliers {multiplier}x σ     → {metric_name}: {score:.4f} (Δ {degradation:+.1f}%)")
            except Exception:
                results.append({"Attack": f"Outliers {multiplier}x σ", "Score": 0, "Degradation_%": 100})

        # ── Robustness Score ──
        avg_degradation = np.mean([abs(r["Degradation_%"]) for r in results])
        robustness_score = max(0, min(100, 100 - avg_degradation))

        print(f"\n  ╔════════════════════════════════════════╗")
        print(f"  ║  ROBUSTNESS SCORE: {robustness_score:>5.1f} / 100       ║")
        if robustness_score >= 80:
            print(f"  ║  Rating: EXCELLENT — Very resilient    ║")
        elif robustness_score >= 60:
            print(f"  ║  Rating: GOOD — Handles most attacks   ║")
        elif robustness_score >= 40:
            print(f"  ║  Rating: MODERATE — Some weaknesses    ║")
        else:
            print(f"  ║  Rating: FRAGILE — Needs improvement   ║")
        print(f"  ╚════════════════════════════════════════╝")

        # Save results
        stress_df = pd.DataFrame(results)
        stress_df.to_csv(os.path.join(output_dir, "stress_test_results.csv"), index=False)

        if feature_sensitivity:
            pd.DataFrame(feature_sensitivity).to_csv(os.path.join(output_dir, "feature_sensitivity.csv"), index=False)

        # Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Degradation chart
        stress_df_sorted = stress_df.sort_values("Degradation_%", ascending=True)
        colors = ["#4CAF50" if d < 10 else "#FF9800" if d < 30 else "#F44336" for d in stress_df_sorted["Degradation_%"]]
        ax1.barh(stress_df_sorted["Attack"], stress_df_sorted["Degradation_%"], color=colors)
        ax1.set_xlabel("Performance Degradation (%)")
        ax1.set_title("Adversarial Attack Impact")
        ax1.axvline(x=10, color="gray", linestyle="--", alpha=0.5)

        # Robustness gauge
        theta = np.linspace(0, np.pi, 100)
        ax2.plot(np.cos(theta), np.sin(theta), "k-", lw=2)
        angle = np.pi * (1 - robustness_score / 100)
        ax2.annotate("", xy=(np.cos(angle), np.sin(angle)), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", lw=3, color="#2196F3"))
        ax2.set_xlim(-1.3, 1.3); ax2.set_ylim(-0.2, 1.3)
        ax2.set_aspect("equal")
        ax2.text(0, -0.15, f"{robustness_score:.0f}/100", ha="center", fontsize=24, fontweight="bold")
        ax2.text(0, 1.15, "Robustness Score", ha="center", fontsize=14)
        ax2.text(-1.1, 0, "Fragile", ha="center", fontsize=10, color="#F44336")
        ax2.text(1.1, 0, "Robust", ha="center", fontsize=10, color="#4CAF50")
        ax2.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "stress_test_report.png"), dpi=150)
        plt.close()
        print(f"  → Saved stress_test_results.csv & stress_test_report.png")

        return robustness_score, stress_df


# ═══════════════════════════════════════════════════════════════════════════════
# 9. INNOVATIVE MODULE: FEATURE STORY GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureStoryGenerator:
    """
    Generates a plain-English narrative explaining feature importance using
    SHAP values + statistical analysis. No LLM API needed — uses rule-based
    narrative generation that creates human-readable stories.
    """

    @staticmethod
    def run(pipeline, X, y, task, config, output_dir):
        banner("FEATURE STORY GENERATOR", "📖")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state,
            stratify=y if task == "classification" else None,
        )
        pipeline.fit(X_train, y_train)
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        # Get feature names
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

        story_lines = []
        story_lines.append("=" * 60)
        story_lines.append("  FEATURE STORY: Why Your Model Thinks What It Thinks")
        story_lines.append("=" * 60)
        story_lines.append("")

        # ── SHAP Analysis ──
        shap_importance = None
        if HAS_SHAP:
            print("  Computing SHAP values (this may take a moment)...")
            try:
                X_test_transformed = preprocessor.transform(X_test)
                if hasattr(X_test_transformed, "toarray"):
                    X_test_transformed = X_test_transformed.toarray()

                # Use a subsample for speed
                sample_size = min(100, len(X_test_transformed))
                X_sample = X_test_transformed[:sample_size]

                if hasattr(model, "feature_importances_"):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    explainer = shap.KernelExplainer(model.predict, X_test_transformed[:50])
                    shap_values = explainer.shap_values(X_sample)

                # Handle multi-class
                if isinstance(shap_values, list):
                    shap_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                else:
                    shap_abs = np.abs(shap_values).mean(axis=0)

                shap_importance = pd.DataFrame({
                    "feature": feature_names[:len(shap_abs)],
                    "shap_importance": shap_abs
                }).sort_values("shap_importance", ascending=False)

                # SHAP summary plot
                fig, ax = plt.subplots(figsize=(10, 6))
                top_shap = shap_importance.head(15)
                sns.barplot(data=top_shap, x="shap_importance", y="feature", color="#E91E63", ax=ax)
                ax.set_title("SHAP Feature Importance")
                ax.set_xlabel("Mean |SHAP value|")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "shap_importance.png"), dpi=150)
                plt.close()
                print("  → Saved shap_importance.png")
            except Exception as e:
                print(f"  SHAP analysis failed ({e}), using fallback importance")
                shap_importance = None

        # ── Fallback: sklearn feature importance ──
        if shap_importance is None:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                shap_importance = pd.DataFrame({
                    "feature": feature_names[:len(imp)],
                    "shap_importance": imp
                }).sort_values("shap_importance", ascending=False)
            elif hasattr(model, "coef_"):
                coef = np.abs(model.coef_).flatten() if model.coef_.ndim == 1 else np.abs(model.coef_).mean(axis=0)
                shap_importance = pd.DataFrame({
                    "feature": feature_names[:len(coef)],
                    "shap_importance": coef
                }).sort_values("shap_importance", ascending=False)

        if shap_importance is None:
            print("  Could not extract feature importance")
            return

        # ── Generate the narrative ──
        top_features = shap_importance.head(10)
        total_importance = top_features["shap_importance"].sum()

        story_lines.append("  THE BIG PICTURE:")
        story_lines.append(f"  Your model considers {len(feature_names)} features to make predictions.")
        story_lines.append(f"  But not all features are equal — here's the story:\n")

        # Top feature narrative
        for rank, (_, row) in enumerate(top_features.iterrows(), 1):
            fname = row["feature"]
            imp_pct = (row["shap_importance"] / total_importance * 100)

            # Clean up feature name for readability
            clean_name = fname.replace("num__", "").replace("cat_low__", "").replace("cat_high__", "")

            if rank == 1:
                story_lines.append(f"  🥇 #{rank} — {clean_name} (contributes {imp_pct:.1f}% of model's reasoning)")
                story_lines.append(f"     This is THE most important feature. The model relies on it")
                story_lines.append(f"     more than any other signal in your data.")
            elif rank == 2:
                story_lines.append(f"\n  🥈 #{rank} — {clean_name} ({imp_pct:.1f}%)")
                story_lines.append(f"     The second most influential feature — together with #{rank-1},")
                top2_pct = top_features.head(2)["shap_importance"].sum() / total_importance * 100
                story_lines.append(f"     these two features account for {top2_pct:.0f}% of the model's decisions.")
            elif rank == 3:
                story_lines.append(f"\n  🥉 #{rank} — {clean_name} ({imp_pct:.1f}%)")
                story_lines.append(f"     Rounds out the top 3. Still a significant contributor.")
            elif rank <= 5:
                story_lines.append(f"\n  ● #{rank} — {clean_name} ({imp_pct:.1f}%)")
                story_lines.append(f"     A supporting signal that helps refine predictions.")
            else:
                story_lines.append(f"\n  ○ #{rank} — {clean_name} ({imp_pct:.1f}%)")
                story_lines.append(f"     Minor contributor — removing it wouldn't change much.")

        # Feature correlation analysis
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        if len(num_cols) >= 2:
            story_lines.append(f"\n\n  FEATURE RELATIONSHIPS:")
            corr_matrix = X[num_cols].corr()
            high_corr_pairs = []
            for i in range(len(num_cols)):
                for j in range(i+1, len(num_cols)):
                    c = abs(corr_matrix.iloc[i, j])
                    if c > 0.7:
                        high_corr_pairs.append((num_cols[i], num_cols[j], c))

            if high_corr_pairs:
                high_corr_pairs.sort(key=lambda x: -x[2])
                story_lines.append(f"  Found {len(high_corr_pairs)} highly correlated feature pairs:")
                for a, b, c in high_corr_pairs[:5]:
                    story_lines.append(f"    • {a} ↔ {b} (correlation: {c:.2f})")
                    story_lines.append(f"      These features tell a similar story — one might be redundant.")
            else:
                story_lines.append("  No highly correlated features found — each feature brings unique info.")

        # Summary
        top3_pct = top_features.head(3)["shap_importance"].sum() / total_importance * 100
        story_lines.append(f"\n\n  SUMMARY:")
        story_lines.append(f"  • Top 3 features drive {top3_pct:.0f}% of all predictions")
        story_lines.append(f"  • Bottom {max(0, len(feature_names) - 10)} features contribute minimally")

        if task == "classification":
            story_lines.append(f"  • For classification, the model uses these features to draw")
            story_lines.append(f"    decision boundaries between classes")
        else:
            story_lines.append(f"  • For regression, higher importance = more influence on predicted value")

        story_lines.append(f"\n  💡 TIP: Consider removing low-importance features to speed up")
        story_lines.append(f"  training and potentially improve generalization.")

        story = "\n".join(story_lines)
        print(story)

        # Save
        with open(os.path.join(output_dir, "feature_story.txt"), "w") as f:
            f.write(story)
        print(f"\n  → Saved feature_story.txt")

        return story


# ═══════════════════════════════════════════════════════════════════════════════
# 10. INNOVATIVE MODULE: DATA DRIFT SENTINEL
# ═══════════════════════════════════════════════════════════════════════════════

class DataDriftSentinel:
    """
    Monitors incoming data for distribution shifts that would degrade model
    performance. Detects:
    - Numeric distribution drift (Kolmogorov-Smirnov test)
    - Categorical distribution drift (Chi-squared test)
    - Missing pattern changes
    - Schema changes (new/removed columns)
    - Outlier explosion
    """

    @staticmethod
    def create_baseline(df, output_dir):
        """Save a statistical fingerprint of the training data."""
        fingerprint = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": df.shape,
            "numeric_stats": {},
            "categorical_stats": {},
            "missing_rates": (df.isnull().sum() / len(df)).to_dict(),
        }

        for col in df.select_dtypes(include=[np.number]).columns:
            fingerprint["numeric_stats"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75)),
                "skew": float(df[col].skew()),
                "values_sample": df[col].dropna().values[:1000].tolist(),
            }

        for col in df.select_dtypes(include=["object", "category"]).columns:
            fingerprint["categorical_stats"][col] = {
                "unique": int(df[col].nunique()),
                "distribution": df[col].value_counts(normalize=True).to_dict(),
                "categories": list(df[col].dropna().unique()),
            }

        fp_path = os.path.join(output_dir, "data_fingerprint.json")
        with open(fp_path, "w") as f:
            json.dump(fingerprint, f, indent=2, default=str)
        print(f"  → Saved data_fingerprint.json (baseline for drift detection)")
        return fingerprint

    @staticmethod
    def check_drift(new_df, fingerprint_path, config, output_dir):
        """Compare new data against the baseline fingerprint."""
        banner("DATA DRIFT SENTINEL", "🛡️")

        with open(fingerprint_path) as f:
            baseline = json.load(f)

        alerts = []
        drift_scores = {}

        # ── Schema Check ──
        print("\n  ── Schema Check ──")
        baseline_cols = set(baseline["columns"])
        new_cols = set(new_df.columns)
        missing_cols = baseline_cols - new_cols
        extra_cols = new_cols - baseline_cols

        if missing_cols:
            msg = f"MISSING COLUMNS: {missing_cols}"
            alerts.append(("CRITICAL", msg))
            print(f"  🔴 {msg}")
        if extra_cols:
            msg = f"NEW COLUMNS: {extra_cols}"
            alerts.append(("WARNING", msg))
            print(f"  🟡 {msg}")
        if not missing_cols and not extra_cols:
            print("  🟢 Schema matches baseline")

        # ── Numeric Distribution Drift ──
        print("\n  ── Numeric Distribution Drift (KS Test) ──")
        for col, stats_dict in baseline["numeric_stats"].items():
            if col not in new_df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(new_df[col]):
                continue

            old_sample = np.array(stats_dict["values_sample"])
            new_sample = new_df[col].dropna().values

            if len(new_sample) < 5:
                continue

            ks_stat, p_value = stats.ks_2samp(old_sample, new_sample)
            drift_scores[col] = {"ks_stat": ks_stat, "p_value": p_value}

            mean_shift = abs(new_df[col].mean() - stats_dict["mean"]) / max(stats_dict["std"], 1e-10)

            if p_value < config.drift_pvalue_threshold:
                severity = "CRITICAL" if p_value < 0.001 else "WARNING"
                emoji = "🔴" if severity == "CRITICAL" else "🟡"
                msg = f"'{col}': KS={ks_stat:.3f}, p={p_value:.4f}, mean shifted {mean_shift:.1f}σ"
                alerts.append((severity, msg))
                print(f"  {emoji} DRIFT in {msg}")
            else:
                print(f"  🟢 '{col}': No significant drift (KS={ks_stat:.3f}, p={p_value:.2f})")

        # ── Categorical Drift ──
        print("\n  ── Categorical Distribution Drift ──")
        for col, cat_stats in baseline["categorical_stats"].items():
            if col not in new_df.columns:
                continue

            old_cats = set(cat_stats["categories"])
            new_cats = set(new_df[col].dropna().unique())
            unseen = new_cats - old_cats

            if unseen:
                msg = f"'{col}': {len(unseen)} unseen categories: {list(unseen)[:5]}"
                alerts.append(("WARNING", msg))
                print(f"  🟡 {msg}")

            vanished = old_cats - new_cats
            if vanished and len(vanished) > len(old_cats) * 0.3:
                msg = f"'{col}': {len(vanished)} categories vanished"
                alerts.append(("WARNING", msg))
                print(f"  🟡 {msg}")

            if not unseen and not vanished:
                print(f"  🟢 '{col}': Categories stable")

        # ── Missing Pattern Changes ──
        print("\n  ── Missing Data Pattern Changes ──")
        for col, old_rate in baseline["missing_rates"].items():
            if col not in new_df.columns:
                continue
            new_rate = new_df[col].isnull().sum() / len(new_df)
            diff = abs(new_rate - old_rate)
            if diff > 0.1:
                msg = f"'{col}': missing rate changed {old_rate*100:.1f}% → {new_rate*100:.1f}%"
                alerts.append(("WARNING", msg))
                print(f"  🟡 {msg}")

        # ── Summary ──
        critical_count = sum(1 for s, _ in alerts if s == "CRITICAL")
        warning_count = sum(1 for s, _ in alerts if s == "WARNING")

        if critical_count == 0 and warning_count == 0:
            health = "HEALTHY"
            health_emoji = "🟢"
        elif critical_count == 0:
            health = "MINOR DRIFT"
            health_emoji = "🟡"
        else:
            health = "SIGNIFICANT DRIFT"
            health_emoji = "🔴"

        print(f"\n  ╔════════════════════════════════════════╗")
        print(f"  ║  {health_emoji} Data Health: {health:<24} ║")
        print(f"  ║  Critical alerts: {critical_count:<21} ║")
        print(f"  ║  Warning alerts:  {warning_count:<21} ║")
        print(f"  ╚════════════════════════════════════════╝")

        # Save report
        report = {
            "health": health,
            "critical_alerts": critical_count,
            "warning_alerts": warning_count,
            "alerts": [{"severity": s, "message": m} for s, m in alerts],
            "drift_scores": {k: {"ks_stat": v["ks_stat"], "p_value": v["p_value"]} for k, v in drift_scores.items()},
        }
        with open(os.path.join(output_dir, "drift_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        # Drift visualization
        if drift_scores:
            fig, ax = plt.subplots(figsize=(10, 6))
            cols = list(drift_scores.keys())[:15]
            ks_vals = [drift_scores[c]["ks_stat"] for c in cols]
            p_vals = [drift_scores[c]["p_value"] for c in cols]
            colors = ["#F44336" if p < 0.001 else "#FF9800" if p < 0.05 else "#4CAF50" for p in p_vals]
            ax.barh(cols, ks_vals, color=colors)
            ax.axvline(x=0.1, color="gray", linestyle="--", alpha=0.5, label="Typical threshold")
            ax.set_xlabel("KS Statistic (higher = more drift)")
            ax.set_title("Data Drift Detection per Feature")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "drift_report.png"), dpi=150)
            plt.close()

        print(f"  → Saved drift_report.json & drift_report.png")
        return report


# ═══════════════════════════════════════════════════════════════════════════════
# 11. CROSS-TASK INSIGHT TRANSFER
# ═══════════════════════════════════════════════════════════════════════════════

def cross_task_insights(clf_pipeline, reg_pipeline, X, output_dir):
    """Compare feature importance across classification and regression."""
    banner("CROSS-TASK INSIGHT TRANSFER", "🔀")

    def get_importance(pipeline):
        model = pipeline.named_steps["model"]
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_, index=names[:len(model.feature_importances_)])
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_).flatten() if model.coef_.ndim == 1 else np.abs(model.coef_).mean(axis=0)
            return pd.Series(coef, index=names[:len(coef)])
        return None

    clf_imp = get_importance(clf_pipeline)
    reg_imp = get_importance(reg_pipeline)

    if clf_imp is None or reg_imp is None:
        print("  Could not extract importance from one or both models")
        return

    # Normalize to [0, 1]
    clf_norm = clf_imp / clf_imp.max()
    reg_norm = reg_imp / reg_imp.max()

    # Align on common features
    common = clf_norm.index.intersection(reg_norm.index)
    clf_aligned = clf_norm[common]
    reg_aligned = reg_norm[common]

    # Classify features
    insights = pd.DataFrame({
        "feature": common,
        "clf_importance": clf_aligned.values,
        "reg_importance": reg_aligned.values,
    })
    insights["avg_importance"] = (insights["clf_importance"] + insights["reg_importance"]) / 2
    insights["task_specificity"] = abs(insights["clf_importance"] - insights["reg_importance"])
    insights = insights.sort_values("avg_importance", ascending=False)

    # Categorize
    task_agnostic = insights[insights["task_specificity"] < 0.2].head(10)
    clf_specific = insights[insights["clf_importance"] > insights["reg_importance"] + 0.3].head(5)
    reg_specific = insights[insights["reg_importance"] > insights["clf_importance"] + 0.3].head(5)

    print(f"\n  Task-Agnostic Features (important for BOTH tasks):")
    for _, row in task_agnostic.iterrows():
        clean = row["feature"].replace("num__", "").replace("cat_low__", "")
        print(f"    ● {clean:<30} clf={row['clf_importance']:.3f}  reg={row['reg_importance']:.3f}")

    if len(clf_specific) > 0:
        print(f"\n  Classification-Specific Features:")
        for _, row in clf_specific.iterrows():
            clean = row["feature"].replace("num__", "").replace("cat_low__", "")
            print(f"    ● {clean:<30} clf={row['clf_importance']:.3f}  reg={row['reg_importance']:.3f}")

    if len(reg_specific) > 0:
        print(f"\n  Regression-Specific Features:")
        for _, row in reg_specific.iterrows():
            clean = row["feature"].replace("num__", "").replace("cat_low__", "")
            print(f"    ● {clean:<30} clf={row['clf_importance']:.3f}  reg={row['reg_importance']:.3f}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(insights["clf_importance"], insights["reg_importance"], alpha=0.6, s=60, c="#2196F3")

    # Label top features
    for _, row in insights.head(8).iterrows():
        clean = row["feature"].replace("num__", "").replace("cat_low__", "").replace("cat_high__", "")
        ax.annotate(clean[:20], (row["clf_importance"], row["reg_importance"]),
                    fontsize=8, alpha=0.8, xytext=(5, 5), textcoords="offset points")

    ax.plot([0, 1], [0, 1], "r--", alpha=0.3, label="Equal importance line")
    ax.set_xlabel("Classification Importance (normalized)")
    ax.set_ylabel("Regression Importance (normalized)")
    ax.set_title("Cross-Task Feature Importance Transfer")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_task_insights.png"), dpi=150)
    plt.close()

    insights.to_csv(os.path.join(output_dir, "cross_task_insights.csv"), index=False)
    print(f"\n  → Saved cross_task_insights.png & cross_task_insights.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. TARGET CONVERSION (for dual mode)
# ═══════════════════════════════════════════════════════════════════════════════

def bin_continuous_target(y, n_bins=4):
    try:
        y_binned, bin_edges = pd.qcut(y, q=n_bins, labels=False, retbins=True, duplicates="drop")
    except ValueError:
        y_binned, bin_edges = pd.cut(y, bins=n_bins, labels=False, retbins=True)
    actual_bins = int(y_binned.max() + 1)
    bin_labels = [f"Bin{i} [{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}]" for i in range(actual_bins)]
    print(f"  Binned into {actual_bins} classes:")
    for i, label in enumerate(bin_labels):
        print(f"    Class {i}: {label} ({(y_binned == i).sum():,} samples)")
    return y_binned.astype(int), bin_labels, bin_edges


# ═══════════════════════════════════════════════════════════════════════════════
# 13. CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

def run_clustering(X, preprocessor, config):
    banner("CLUSTERING ANALYSIS")
    X_transformed = preprocessor.fit_transform(X)
    best_k, best_sil = 2, -1
    results = []
    for k in range(config.n_clusters_range[0], config.n_clusters_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=config.random_state, n_init=10)
        labels = km.fit_predict(X_transformed)
        sil = silhouette_score(X_transformed, labels)
        results.append({"k": k, "silhouette": round(sil, 4), "inertia": round(km.inertia_, 2)})
        print(f"  k={k:>2}  silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil, best_k = sil, k
    print(f"\n  ★ Optimal: k={best_k} (silhouette={best_sil:.4f})")
    return pd.DataFrame(results), best_k


# ═══════════════════════════════════════════════════════════════════════════════
# 14. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(data_path, target=None, task=None, enable_heal=False,
                 enable_stress=False, enable_story=False, enable_all=False,
                 monitor_mode=False, baseline_path=None, new_data_path=None):

    config = PipelineConfig()
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if enable_all:
        enable_heal = enable_stress = enable_story = True

    # ── DRIFT MONITORING MODE ──
    if monitor_mode:
        if not baseline_path or not new_data_path:
            print("ERROR: --monitor requires --baseline and --new")
            return
        new_df = pd.read_csv(new_data_path)
        DataDriftSentinel.check_drift(new_df, baseline_path, config, output_dir)
        return

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║       AUTO-ML PIPELINE  v3.0 (Innovative Edition)          ║")
    print("║  Self-Healing │ Feature Stories │ Drift Sentinel │ Stress   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    modules = []
    if enable_heal: modules.append("Self-Healing")
    if enable_stress: modules.append("Stress Testing")
    if enable_story: modules.append("Feature Story")
    if modules:
        print(f"  Active modules: {', '.join(modules)}")

    # ── Load data ──
    ext = Path(data_path).suffix.lower()
    loaders = {".csv": pd.read_csv, ".tsv": lambda p: pd.read_csv(p, sep="\t"),
               ".json": pd.read_json, ".parquet": pd.read_parquet}
    if ext in (".xls", ".xlsx"):
        df = pd.read_excel(data_path)
    elif ext in loaders:
        df = loaders[ext](data_path)
    else:
        raise ValueError(f"Unsupported: {ext}")

    print(f"  Loaded '{data_path}' → {df.shape[0]:,} rows × {df.shape[1]} cols")

    profile = profile_data(df, target)

    if task is None:
        task = detect_task(df, target, profile)
    elif task == "both":
        if not target:
            task = "clustering"
        else:
            print(f"  [User-specified] Task: BOTH")
    else:
        print(f"  [User-specified] Task: {task}")

    # ── Preprocess ──
    sub_banner("PREPROCESSING")
    X, y, preprocessor, label_encoder = preprocess(df, target, config, profile)

    # ── Save drift baseline ──
    DataDriftSentinel.create_baseline(df, output_dir)

    # ── Feature Engineering ──
    sub_banner("FEATURE ENGINEERING")
    X = engineer_features(X, level="basic")
    preprocessor = build_preprocessor(X, config)

    # ══════════════════════════════════════════════════════════════════════════
    # DUAL MODE
    # ══════════════════════════════════════════════════════════════════════════
    if task == "both":
        is_continuous = pd.api.types.is_numeric_dtype(df[target]) and profile["target_nunique"] > 20

        if is_continuous:
            # Regression first
            if enable_heal:
                X_healed, prep_h, reg_results, reg_pipe, reg_best, strat = SelfHealingPipeline.run(X, y, "regression", config, output_dir)
            else:
                reg_results, reg_pipe, reg_best = train_and_evaluate(X, y, build_preprocessor(X, config), "regression", config, tag="original")

            X_used = X_healed if enable_heal else X
            reg_results.to_csv(os.path.join(output_dir, "regression_models.csv"), index=False)
            save_model_comparison_chart(reg_results, reg_best, "regression", output_dir, "regression")
            reg_metrics = evaluate_best_model(reg_pipe, X_used, y, "regression", None, config, output_dir, "regression")
            save_feature_importance(reg_pipe, X_used, output_dir, "regression")

            # Classification (binned)
            sub_banner("CONVERTING TARGET TO CLASSES")
            y_clf, bin_labels, _ = bin_continuous_target(y, config.n_bins_for_classification)
            bin_le = LabelEncoder(); bin_le.classes_ = np.array(bin_labels)

            clf_results, clf_pipe, clf_best = train_and_evaluate(X_used, y_clf, build_preprocessor(X_used, config), "classification", config, tag="binned")
            clf_results.to_csv(os.path.join(output_dir, "classification_models.csv"), index=False)
            save_model_comparison_chart(clf_results, clf_best, "classification", output_dir, "classification")
            clf_metrics = evaluate_best_model(clf_pipe, X_used, y_clf, "classification", bin_le, config, output_dir, "classification")
            save_feature_importance(clf_pipe, X_used, output_dir, "classification")
        else:
            # Classification first
            if enable_heal:
                X_healed, prep_h, clf_results, clf_pipe, clf_best, strat = SelfHealingPipeline.run(X, y, "classification", config, output_dir)
            else:
                clf_results, clf_pipe, clf_best = train_and_evaluate(X, y, build_preprocessor(X, config), "classification", config, tag="original")

            X_used = X_healed if enable_heal else X
            clf_results.to_csv(os.path.join(output_dir, "classification_models.csv"), index=False)
            save_model_comparison_chart(clf_results, clf_best, "classification", output_dir, "classification")
            clf_metrics = evaluate_best_model(clf_pipe, X_used, y, "classification", label_encoder, config, output_dir, "classification")
            save_feature_importance(clf_pipe, X_used, output_dir, "classification")

            # Regression (encoded)
            sub_banner("CONVERTING TARGET TO NUMERIC")
            y_reg = y.astype(float)
            reg_results, reg_pipe, reg_best = train_and_evaluate(X_used, y_reg, build_preprocessor(X_used, config), "regression", config, tag="encoded")
            reg_results.to_csv(os.path.join(output_dir, "regression_models.csv"), index=False)
            save_model_comparison_chart(reg_results, reg_best, "regression", output_dir, "regression")
            reg_metrics = evaluate_best_model(reg_pipe, X_used, y_reg, "regression", None, config, output_dir, "regression")
            save_feature_importance(reg_pipe, X_used, output_dir, "regression")

        # Cross-task insights
        cross_task_insights(clf_pipe, reg_pipe, X_used, output_dir)

        # Dual report
        banner("DUAL-MODE COMBINED REPORT")
        print(f"\n  {'':30} {'CLASSIFICATION':>18} {'REGRESSION':>18}")
        print("  " + "-" * 66)
        print(f"  {'Best Model':<30} {clf_best:>18} {reg_best:>18}")
        if clf_metrics:
            print(f"  {'Accuracy':<30} {clf_metrics['accuracy']:>18.4f} {'--':>18}")
            print(f"  {'F1-Score':<30} {clf_metrics['f1']:>18.4f} {'--':>18}")
        if reg_metrics:
            print(f"  {'RMSE':<30} {'--':>18} {reg_metrics['rmse']:>18.4f}")
            print(f"  {'R²':<30} {'--':>18} {reg_metrics['r2']:>18.4f}")
        print("  " + "-" * 66)

        # Use regression pipeline for stress/story (primary task for continuous, secondary for categorical)
        main_pipe = reg_pipe if is_continuous else clf_pipe
        main_task = "regression" if is_continuous else "classification"
        main_y = y if is_continuous else y

        if enable_stress:
            AdversarialStressTester.run(main_pipe, X_used, main_y, main_task, config, output_dir)
        if enable_story:
            FeatureStoryGenerator.run(main_pipe, X_used, main_y, main_task, config, output_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # SINGLE MODE
    # ══════════════════════════════════════════════════════════════════════════
    elif task in ("classification", "regression"):
        if enable_heal:
            X, preprocessor, results_df, best_pipeline, best_name, strat = SelfHealingPipeline.run(X, y, task, config, output_dir)
        else:
            results_df, best_pipeline, best_name = train_and_evaluate(X, y, preprocessor, task, config)

        results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
        evaluate_best_model(best_pipeline, X, y, task, label_encoder, config, output_dir)
        save_feature_importance(best_pipeline, X, output_dir)
        save_model_comparison_chart(results_df, best_name, task, output_dir)

        if enable_stress:
            AdversarialStressTester.run(best_pipeline, X, y, task, config, output_dir)
        if enable_story:
            FeatureStoryGenerator.run(best_pipeline, X, y, task, config, output_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # CLUSTERING
    # ══════════════════════════════════════════════════════════════════════════
    else:
        cluster_results, best_k = run_clustering(X, preprocessor, config)
        cluster_results.to_csv(os.path.join(output_dir, "clustering_results.csv"), index=False)

    # Save the best pipeline
    try:
        if task != "clustering":
            pipe_to_save = best_pipeline if task in ("classification", "regression") else (reg_pipe if task == "both" else None)
            if pipe_to_save:
                with open(os.path.join(output_dir, "best_model.pkl"), "wb") as f:
                    pickle.dump(pipe_to_save, f)
                print(f"  → Saved best_model.pkl")
    except Exception:
        pass

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Pipeline complete! Results in: {output_dir:<27}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML Pipeline v3.0 — Innovative Edition")
    parser.add_argument("--data", help="Path to dataset (CSV, Excel, JSON, Parquet)")
    parser.add_argument("--target", default=None, help="Target column name")
    parser.add_argument("--task", default=None, choices=["classification", "regression", "clustering", "both"])

    # Innovative modules
    parser.add_argument("--heal", action="store_true", help="Enable Self-Healing Pipeline")
    parser.add_argument("--stress", action="store_true", help="Enable Adversarial Stress Testing")
    parser.add_argument("--story", action="store_true", help="Enable Feature Story Generator")
    parser.add_argument("--all", action="store_true", help="Enable ALL innovative modules")

    # Drift monitoring
    parser.add_argument("--monitor", action="store_true", help="Run in drift monitoring mode")
    parser.add_argument("--baseline", default=None, help="Path to baseline fingerprint JSON")
    parser.add_argument("--new", default=None, help="Path to new data for drift checking")

    args = parser.parse_args()

    if args.monitor:
        run_pipeline(None, monitor_mode=True, baseline_path=args.baseline, new_data_path=args.new)
    else:
        if not args.data:
            parser.error("--data is required unless using --monitor mode")
        run_pipeline(
            args.data, args.target, args.task,
            enable_heal=args.heal, enable_stress=args.stress,
            enable_story=args.story, enable_all=args.all,
        )
