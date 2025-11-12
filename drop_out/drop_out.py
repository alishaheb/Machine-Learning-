# import all dependencies for preprocessing and modeling of drop out dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# =========================================================
# 1. Load data
df = pd.read_csv("drop_out_data.csv")
# 2. Drop useless columns
df = df.drop(columns=["Student_ID", "Enrollment_ID"])
# 3. Target and features
y = df["Dropped_Out"]
X = df.drop(columns=["Dropped_Out"])
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
