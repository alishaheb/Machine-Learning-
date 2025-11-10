# preprocessing data for survival analysis
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# =========================================================
# 1. Load data
# =========================================================
df = pd.read_csv("Hp treatment training.csv", encoding="latin1")

# 2. Drop useless columns
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

# 3. Normalize text / features
df["Treated_with_drugs"] = df["Treated_with_drugs"].str.upper()

# Target and features
y = df["Survived_1_year"]
X = df.drop(columns=["Survived_1_year", "ID_Patient_Care_Situation", "Patient_ID"])

categorical_cols = [
    "Treated_with_drugs",
    "Patient_Smoker",
    "Patient_Rural_Urban",
    "Patient_mental_condition",
]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# =========================================================
# 4. Train/test split (we'll do CV on the training part)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# 5. Preprocessing pipelines
# =========================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    # sparse output is fine; we densify only for Naive Bayes later
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# =========================================================
# 6. Models (as pipelines)
# =========================================================
dt_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", DecisionTreeClassifier(random_state=42))
])

lr_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, solver="lbfgs"))
])

mlp_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
])

# Naive Bayes needs dense input, so we add a to_dense step
nave_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("to_dense", FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
    ("model", GaussianNB())
])

xgb_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_estimators=300
    ))
])

models = {
    "Decision Tree": dt_clf,
    "Logistic Regression": lr_clf,
    "MLP Classifier": mlp_clf,
    "Naive Bayes Classifier": nave_clf,
    "XGBoost Classifier": xgb_clf,
}

# =========================================================
# 7. Cross-validation on the training data
# =========================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

print("===== 5-fold Cross-Validation Results (on training data) =====")

for name, model in models.items():
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    print(f"\n=== {name} ===")
    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        print(f"{metric.capitalize():<9}: mean = {scores.mean():.4f}, std = {scores.std():.4f}")

# =========================================================
# 8. Fit on full training set and evaluate on test set
# =========================================================
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} (Test set) ===")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 score :", f1)
    print("ROC AUC  :", roc_auc)
    print("Confusion matrix:\n", cm)

print("\n===== Final Evaluation on Test Set =====")

for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(name, model, X_test, y_test)
