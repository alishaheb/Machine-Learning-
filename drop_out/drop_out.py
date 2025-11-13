# import all dependencies for preprocessing and modeling of drop out dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import warnings
# =========================================================
# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# =========================================================
# 1. Load data
df = pd.read_csv("dropout.csv", sep=";")

x = df.head(5)
print(x)
# 2. Drop useless columns
# Turn Target into binary: 1 = Dropout, 0 = otherwise
df["Target"] = (df["Target"] == "Dropout").astype(int)

# Quick check
print(df["Target"].value_counts())

# 3. Target and features
y = df["Target"]
X = df.drop(columns=["Target"])

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
# =========================================================
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# =========================================================
# 5. Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)
# Now `preprocess` can be used in model pipelines
# =========================================================
# 6. Models (as pipelines)
from sklearn.tree import DecisionTreeClassifier
dt_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", DecisionTreeClassifier(random_state=42))
])
from sklearn.ensemble import RandomForestClassifier
rf_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(random_state=42))
])
from sklearn.linear_model import LogisticRegression
lr_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])
from sklearn.svm import SVC
svm_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", SVC(random_state=42))
])


from sklearn.neural_network import MLPClassifier
mlp_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", MLPClassifier(max_iter=500, random_state=42))
])
from sklearn.naive_bayes import GaussianNB
gnb_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", GaussianNB())
])
from sklearn.neighbors import KNeighborsClassifier
knn_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", KNeighborsClassifier())
])
from xgboost import XGBClassifier
xgb_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", XGBClassifier(
        objective="binary:logistic",
        random_state=42
    ))
])
models = {
    "Decision Tree": dt_clf,
    "Random Forest": rf_clf,
    "Logistic Regression": lr_clf,
    "SVM": svm_clf,
    "MLP Classifier": mlp_clf,
    "Gaussian NB": gnb_clf,
    "KNN": knn_clf,
    "XGBoost Classifier": xgb_clf,
}
# Models can now be trained and evaluated using `X_train`, `X_test`, `y_train`, and `y_test`
# }
# Models can now be trained and evaluated using `X_train`, `X_test`, `y_train`, and `y_test`
# }
# =========================================================
#evlaluation like f1 score, accuracy, precision, recall
from sklearn.model_selection import cross_validate, StratifiedKFold
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
        return_train_score=False
    )
    print(f"\nModel: {name}")
    for metric in scoring.keys():
        mean_score = cv_results[f'test_{metric}'].mean()
        std_score = cv_results[f'test_{metric}'].std()
        print(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")
# =========================================================
# 7. Final evaluation on test data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
print("===== Final Evaluation on Test Data =====")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model.named_steps['model'], "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    roc = roc_auc_score(y_test, y_proba[:, 1]) if y_proba is not None else None

    print(f"\nModel: {name}")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    if roc is not None:
        print(f"ROC AUC  : {roc:.4f}")

