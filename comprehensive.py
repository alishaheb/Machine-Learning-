"""
AutoML Pipeline for Tabular Data
=================================
A universal machine learning pipeline that automatically:
  1. Profiles & understands your dataset
  2. Cleans and preprocesses data
  3. Engineers features
  4. Detects the task type (classification / regression)
  5. Trains & compares multiple models
  6. Evaluates and generates a summary report

Usage:
    python auto_ml_pipeline.py --data your_data.csv --target target_column
    python auto_ml_pipeline.py --data your_data.csv --target target_column --task classification
    python auto_ml_pipeline.py --data your_data.csv  # (unsupervised / clustering mode)

Requirements:
    pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
"""

import argparse
import os
import sys
import json
import warnings
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline

# Classification models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Regression models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Optional: XGBoost and LightGBM
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

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Central configuration for the pipeline."""

    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    max_onehot_cardinality: int = 15  # one-hot encode if unique values <= this
    missing_threshold: float = 0.5  # drop column if >50% missing
    correlation_threshold: float = 0.95  # drop near-duplicate features
    n_clusters_range: tuple = (2, 10)  # range for auto-cluster search
    output_dir: str = "pipeline_output"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Profiling
# ─────────────────────────────────────────────────────────────────────────────

def profile_data(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    """Generate a comprehensive data profile."""
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
        if df[target].dtype in [np.number] or pd.api.types.is_numeric_dtype(df[target]):
            profile["target_stats"] = df[target].describe().to_dict()
        else:
            profile["target_distribution"] = df[target].value_counts().to_dict()

    print("\n" + "=" * 60)
    print("  DATA PROFILE")
    print("=" * 60)
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

    print("=" * 60)
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# 2. Task Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_task(df: pd.DataFrame, target: Optional[str], profile: dict) -> str:
    """Auto-detect whether this is classification, regression, or clustering."""
    if target is None or target not in df.columns:
        print("  [Auto-detect] No target column → Clustering")
        return "clustering"

    col = df[target]
    nunique = profile["target_nunique"]

    # If object/category → classification
    if col.dtype == "object" or col.dtype.name == "category":
        print(f"  [Auto-detect] Categorical target ({nunique} classes) → Classification")
        return "classification"

    # If numeric with few unique values → classification
    if pd.api.types.is_numeric_dtype(col) and nunique <= 20:
        print(f"  [Auto-detect] Numeric target with {nunique} unique values → Classification")
        return "classification"

    print(f"  [Auto-detect] Continuous target ({nunique} unique values) → Regression")
    return "regression"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    target: Optional[str],
    config: PipelineConfig,
    profile: dict,
) -> tuple:
    """
    Clean and preprocess the data:
      - Drop high-missing columns
      - Parse datetime features
      - Encode target (if classification)
      - Split features / target
      - Build sklearn ColumnTransformer
    """
    df = df.copy()

    # ── Drop columns that are mostly missing ──
    drop_cols = [
        c
        for c, pct in profile["missing_pct"].items()
        if pct > config.missing_threshold * 100
    ]
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} high-missing columns: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)

    # ── Drop constant columns ──
    const_cols = [c for c in df.columns if df[c].nunique() <= 1 and c != target]
    if const_cols:
        print(f"  Dropping {len(const_cols)} constant columns: {const_cols}")
        df.drop(columns=const_cols, inplace=True)

    # ── Parse datetime → numeric features ──
    for col in profile["datetime_cols"]:
        if col in df.columns and col != target:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df.drop(columns=[col], inplace=True)
            print(f"  Expanded datetime '{col}' → year/month/day/dayofweek")

    # ── Also try to parse object cols that look like dates ──
    for col in list(df.select_dtypes(include=["object"]).columns):
        if col == target:
            continue
        sample = df[col].dropna().head(50)
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True)
            # If >80% parsed successfully, treat as datetime
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df.drop(columns=[col], inplace=True)
            print(f"  Detected & expanded date column '{col}'")
        except (ValueError, TypeError):
            pass

    # ── Drop ID-like columns (high cardinality + unique ratio > 0.9) ──
    for col in list(df.select_dtypes(include=["object"]).columns):
        if col == target:
            continue
        if df[col].nunique() / max(len(df), 1) > 0.9:
            print(f"  Dropping ID-like column: '{col}'")
            df.drop(columns=[col], inplace=True)

    # ── Encode target ──
    label_encoder = None
    if target and target in df.columns:
        y = df[target]
        X = df.drop(columns=[target])
        if y.dtype == "object" or y.dtype.name == "category":
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y.astype(str)), name=target)
            print(f"  Encoded target classes: {list(label_encoder.classes_)}")
    else:
        X = df
        y = None

    # ── Identify final column types ──
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)

    low_card_cat = [c for c in cat_cols if X[c].nunique() <= config.max_onehot_cardinality]
    high_card_cat = [c for c in cat_cols if X[c].nunique() > config.max_onehot_cardinality]

    # ── Build ColumnTransformer ──
    transformers = []

    if num_cols:
        num_pipeline = SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if low_card_cat:
        cat_pipeline = SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat_low", cat_pipeline, low_card_cat))

    if high_card_cat:
        ord_pipeline = SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("cat_high", ord_pipeline, high_card_cat))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    print(f"\n  Final features: {len(num_cols)} numeric, {len(low_card_cat)} low-card cat, {len(high_card_cat)} high-card cat")

    return X, y, preprocessor, label_encoder


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight derived features for numeric columns."""
    X = X.copy()
    num_cols = list(X.select_dtypes(include=[np.number]).columns)

    if len(num_cols) >= 2:
        # Pairwise ratios for top correlated pairs (limit to avoid explosion)
        corr = X[num_cols].corr().abs()
        pairs_added = 0
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                if corr.iloc[i, j] > 0.5 and pairs_added < 5:
                    a, b = num_cols[i], num_cols[j]
                    X[f"{a}_div_{b}"] = X[a] / X[b].replace(0, np.nan)
                    pairs_added += 1

    # Log-transform skewed features
    for col in num_cols:
        skew = X[col].skew()
        if abs(skew) > 2 and (X[col] > 0).all():
            X[f"{col}_log"] = np.log1p(X[col])

    added = len(X.columns) - len(num_cols) - len(
        X.select_dtypes(include=["object", "category"]).columns
    )
    if added > 0:
        print(f"  Engineered {added} new features")

    return X


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model Selection & Training
# ─────────────────────────────────────────────────────────────────────────────

def get_models(task: str) -> dict:
    """Return a dictionary of candidate models for the detected task."""
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
            models["XGBoost"] = XGBClassifier(
                n_estimators=200, use_label_encoder=False,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
        if HAS_LGBM:
            models["LightGBM"] = LGBMClassifier(
                n_estimators=200, random_state=42, verbose=-1,
            )

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
            models["XGBoost"] = XGBRegressor(
                n_estimators=200, random_state=42, verbosity=0,
            )
        if HAS_LGBM:
            models["LightGBM"] = LGBMRegressor(
                n_estimators=200, random_state=42, verbose=-1,
            )

    else:  # clustering
        models = {
            "KMeans": None,  # handled separately
            "Agglomerative": None,
            "DBSCAN": None,
        }

    return models


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    task: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Train all candidate models with cross-validation and return results."""
    models = get_models(task)

    if task == "classification":
        scoring = "f1_weighted"
        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    else:
        scoring = "neg_mean_squared_error"
        cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

    results = []
    best_score = -np.inf
    best_name = None
    best_pipeline = None

    print("\n" + "=" * 60)
    print("  MODEL TRAINING & CROSS-VALIDATION")
    print("=" * 60)
    print(f"  {'Model':<22} {'CV Score':>12} {'Std':>10}  {'Time':>8}")
    print("  " + "-" * 56)

    for name, model in models.items():
        pipe = SkPipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        start = time.time()
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            elapsed = time.time() - start
            mean_score = scores.mean()
            std_score = scores.std()

            display_score = mean_score if task == "classification" else -mean_score

            results.append({
                "Model": name,
                "CV_Score_Mean": mean_score,
                "CV_Score_Std": std_score,
                "Display_Score": display_score,
                "Time_s": round(elapsed, 2),
            })

            label = "F1-weighted" if task == "classification" else "RMSE"
            print(f"  {name:<22} {display_score:>12.4f} {std_score:>10.4f}  {elapsed:>7.1f}s")

            if mean_score > best_score:
                best_score = mean_score
                best_name = name
                best_pipeline = pipe

        except Exception as e:
            print(f"  {name:<22} {'FAILED':>12}  ({e})")

    print("  " + "-" * 56)
    metric_label = "F1-weighted" if task == "classification" else "RMSE"
    best_display = best_score if task == "classification" else -best_score
    print(f"  ★ Best: {best_name} ({metric_label} = {best_display:.4f})")

    # Refit best model on full training data
    best_pipeline.fit(X, y)

    results_df = pd.DataFrame(results).sort_values("CV_Score_Mean", ascending=False)
    return results_df, best_pipeline, best_name


def run_clustering(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Run clustering and find the best number of clusters."""
    print("\n" + "=" * 60)
    print("  CLUSTERING ANALYSIS")
    print("=" * 60)

    X_transformed = preprocessor.fit_transform(X)

    # Find optimal k via silhouette score
    best_k = 2
    best_sil = -1
    results = []

    for k in range(config.n_clusters_range[0], config.n_clusters_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=config.random_state, n_init=10)
        labels = km.fit_predict(X_transformed)
        sil = silhouette_score(X_transformed, labels)
        results.append({"k": k, "silhouette": round(sil, 4), "inertia": round(km.inertia_, 2)})
        print(f"  k={k:>2}  silhouette={sil:.4f}  inertia={km.inertia_:,.0f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k

    print(f"\n  ★ Optimal clusters: k={best_k} (silhouette={best_sil:.4f})")

    # Also try DBSCAN
    try:
        db = DBSCAN(eps=0.5, min_samples=5)
        db_labels = db.fit_predict(X_transformed)
        n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        if n_clusters_db >= 2:
            db_sil = silhouette_score(X_transformed, db_labels)
            print(f"  DBSCAN found {n_clusters_db} clusters (silhouette={db_sil:.4f})")
    except Exception:
        pass

    return pd.DataFrame(results), best_k


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation & Reporting
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_best_model(
    pipeline: SkPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    label_encoder: Optional[LabelEncoder],
    config: PipelineConfig,
    output_dir: str,
):
    """Full evaluation of the best model on a held-out test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state,
        stratify=y if task == "classification" else None,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n" + "=" * 60)
    print("  HOLDOUT TEST EVALUATION")
    print("=" * 60)

    if task == "classification":
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"  Accuracy:   {acc:.4f}")
        print(f"  F1-Score:   {f1:.4f}")
        print(f"  Precision:  {prec:.4f}")
        print(f"  Recall:     {rec:.4f}")

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        if label_encoder:
            labels = label_encoder.classes_
        else:
            labels = sorted(y.unique())
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
        plt.close()
        print(f"  → Saved confusion_matrix.png")

    else:  # regression
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  RMSE:  {rmse:.4f}")
        print(f"  MAE:   {mae:.4f}")
        print(f"  R²:    {r2:.4f}")

        # Actual vs Predicted plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.4, s=20)
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect prediction")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted (R² = {r2:.3f})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=150)
        plt.close()
        print(f"  → Saved actual_vs_predicted.png")

    # Feature importance (if available)
    try:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            pre = pipeline.named_steps["preprocessor"]
            feature_names = pre.get_feature_names_out()
            imp_df = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(20)
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, color="steelblue")
            ax.set_title("Top 20 Feature Importances")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
            plt.close()
            print(f"  → Saved feature_importance.png")

            imp_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(data_path: str, target: Optional[str] = None, task: Optional[str] = None):
    """Execute the full AutoML pipeline."""
    config = PipelineConfig()
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║             AUTO-ML PIPELINE  v1.0                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Load data ──
    ext = Path(data_path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(data_path)
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(data_path)
    elif ext == ".json":
        df = pd.read_json(data_path)
    elif ext == ".parquet":
        df = pd.read_parquet(data_path)
    elif ext == ".tsv":
        df = pd.read_csv(data_path, sep="\t")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    print(f"  Loaded '{data_path}' → {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── Profile ──
    profile = profile_data(df, target)

    # ── Detect task ──
    if task is None:
        task = detect_task(df, target, profile)
    else:
        print(f"  [User-specified] Task: {task}")

    # ── Preprocess ──
    print("\n── PREPROCESSING ──")
    X, y, preprocessor, label_encoder = preprocess(df, target, config, profile)

    # ── Feature engineering ──
    print("\n── FEATURE ENGINEERING ──")
    X = engineer_features(X)

    # Re-detect column types after engineering
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
    low_card = [c for c in cat_cols if X[c].nunique() <= config.max_onehot_cardinality]
    high_card = [c for c in cat_cols if X[c].nunique() > config.max_onehot_cardinality]

    transformers = []
    if num_cols:
        transformers.append(("num", SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols))
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

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # ── Train & Evaluate ──
    if task in ("classification", "regression"):
        results_df, best_pipeline, best_name = train_and_evaluate(X, y, preprocessor, task, config)

        # Save results
        results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
        print(f"\n  → Saved model_comparison.csv")

        # Detailed evaluation of the best model
        evaluate_best_model(best_pipeline, X, y, task, label_encoder, config, output_dir)

        # Model comparison chart
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_label = "F1-Weighted" if task == "classification" else "RMSE"
        results_df_sorted = results_df.sort_values("Display_Score", ascending=True)
        colors = ["#2196F3" if m != best_name else "#FF9800" for m in results_df_sorted["Model"]]
        ax.barh(results_df_sorted["Model"], results_df_sorted["Display_Score"], color=colors)
        ax.set_xlabel(metric_label)
        ax.set_title(f"Model Comparison ({metric_label})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150)
        plt.close()
        print(f"  → Saved model_comparison.png")

    else:
        cluster_results, best_k = run_clustering(X, preprocessor, config)
        cluster_results.to_csv(os.path.join(output_dir, "clustering_results.csv"), index=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(cluster_results["k"], cluster_results["silhouette"], "bo-")
        ax1.axvline(x=best_k, color="r", linestyle="--", label=f"Best k={best_k}")
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Silhouette Score")
        ax1.set_title("Silhouette Analysis")
        ax1.legend()

        ax2.plot(cluster_results["k"], cluster_results["inertia"], "go-")
        ax2.set_xlabel("Number of Clusters (k)")
        ax2.set_ylabel("Inertia")
        ax2.set_title("Elbow Method")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "clustering_analysis.png"), dpi=150)
        plt.close()
        print(f"  → Saved clustering_analysis.png")

    print("\n╔══════════════════════════════════════════════════════════╗")
    print(f"║  Pipeline complete! Results saved to: {output_dir:<19}║")
    print("╚══════════════════════════════════════════════════════════╝\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML Pipeline for Tabular Data")
    parser.add_argument("--data", required=True, help="Path to the dataset (CSV, Excel, JSON, Parquet)")
    parser.add_argument("--target", default=None, help="Target column name (omit for clustering)")
    parser.add_argument("--task", default=None, choices=["classification", "regression", "clustering"],
                        help="Force a specific task type (auto-detected if omitted)")
    args = parser.parse_args()

    run_pipeline(args.data, args.target, args.task)