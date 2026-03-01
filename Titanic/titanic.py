#preprocessing titanic dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# =========================================================
# 1. Load data
df = pd.read_csv("Titanic-Dataset.csv")
# 2. Drop useless columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
# 3. Target and features
y = df["Survived"]
X = df.drop(columns=["Survived"])
# Identify categorical and numeric columns
dfategorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
dnumeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
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
        ("num", numeric_transformer, dnumeric_cols),
        ("cat", categorical_transformer, dfategorical_cols),
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
# Similarly, other models can be defined using the `preprocess` pipeline
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
    ("model", SVC(probability=True, random_state=42))
])
from sklearn.neighbors import KNeighborsClassifier
knn_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", KNeighborsClassifier())
])
from sklearn.naive_bayes import GaussianNB
gnb_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", GaussianNB())
])
from xgboost import XGBClassifier
xgb_clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    ))
])
models = {
    "Decision Tree": dt_clf,
    "Random Forest": rf_clf,
    "Logistic Regression": lr_clf,
    "SVM": svm_clf,
    "KNN": knn_clf,
    "Gaussian NB": gnb_clf,
    "XGBoost Classifier": xgb_clf,
}
# Models can now be trained and evaluated using `X_train`, `X_test`, `y_train`, and `y_test`
# }
# Models can now be trained and evaluated using `X_train`, `X_test`, `y_train`, and `y_test`
# }
# =========================================================
# 7. Cross-validation on the training data (baseline models)
from sklearn.model_selection import StratifiedKFold, cross_validate
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
    print(f"Model: {name}")
    for metric in scoring.keys():
        mean_score = cv_results[f'test_{metric}'].mean()
        std_score = cv_results[f'test_{metric}'].std()
        print(f"  {metric}: {mean_score:.4f} ± {std_score:.4f}")
    print("------------------------------------------------------------")
# =========================================================
# 8. Final evaluation on the test data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
print("===== Final Evaluation on Test Data =====")
for name, model in models.items():
    # Train the model on the entire training set
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A'
    # Print results
    print(f"Model: {name}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc if roc_auc == 'N/A' else f'{roc_auc:.4f}'}")
    print("------------------------------------------------------------")
# =========================================================
# 9. Hyperparameter tuning for XGBoost (GridSearchCV)
from sklearn.model_selection import GridSearchCV
param_grid_xgb = {
    "model__n_estimators": [50, 100, 150],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1, 0.2],
}
grid_xgb = GridSearchCV(
    xgb_clf,
    param_grid=param_grid_xgb,
    scoring="roc_auc",   # main metric for tuning
    cv=cv,
    n_jobs=-1
)
print("\n===== Tuning XGBoost with GridSearchCV =====")
grid_xgb.fit(X_train, y_train)
print("Best XGBoost params:", grid_xgb.best_params_)
print("Best XGBoost CV ROC AUC:", grid_xgb.best_score_)
best_xgb = grid_xgb.best_estimator_
# Final evaluation of tuned XGBoost on test set
y_pred = best_xgb.predict(X_test)
y_proba = best_xgb.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
print("\n=== Tuned XGBoost Classifier (Test set) ===")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 score :", f1)
print("ROC AUC  :", roc_auc)
from sklearn.metrics import confusion_matrix
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
# =========================================================
# End of Titanic/titanic.py


