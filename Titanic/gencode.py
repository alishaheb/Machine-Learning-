import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error
)

from xgboost import XGBClassifier   # pip install xgboost


class MLProcessor:
    def __init__(self, file_path: str, target_column: str):
        self.file_path = file_path
        self.target_column = target_column

        self.df = pd.read_csv(file_path)
        self.X = None
        self.y = None

        self.task_type = None
        self.preprocessor = None
        self.numeric_features = None
        self.categorical_features = None

        # train/test
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # current model
        self.model = None

    def preprocess(self):
        print("Starting preprocessing...")
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(axis=1, how='all', inplace=True)

        # Drop irrelevant / ID columns (Titanic-specific)
        drop_cols = [col for col in ['PassengerId', 'Name', 'Ticket', 'Cabin']
                     if col in self.df.columns]
        if drop_cols:
            print(f"Dropping columns: {drop_cols}")
            self.df.drop(columns=drop_cols, inplace=True)

        # Separate target and features
        self.y = self.df[self.target_column]
        self.X = self.df.drop(columns=[self.target_column])

        # Determine task type
        if ((self.y.nunique() <= 20 and self.y.dtype == 'object') or self.y.dtype == 'bool'):
            self.task_type = 'classification'
        elif self.y.nunique() <= 2:
            self.task_type = 'classification'
        elif self.y.dtype in ['int64', 'float64'] and self.y.nunique() > 20:
            self.task_type = 'regression'
        else:
            raise ValueError("Unable to determine task type. Please check the target column.")

        print(f"Detected task type: {self.task_type}")

        # Identify feature types
        self.numeric_features = self.X.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        self.categorical_features = self.X.select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()

        print(f"Numeric features: {self.numeric_features}")
        print(f"Categorical features: {self.categorical_features}")

        # Pipelines
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # NOTE: new sklearn uses sparse_output instead of sparse
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, self.numeric_features),
            ('cat', categorical_pipeline, self.categorical_features)
        ])

        # Fit + transform X
        self.X = self.preprocessor.fit_transform(self.X)

        # Encode y for classification if needed
        if self.task_type == 'classification' and not pd.api.types.is_integer_dtype(self.y):
            print("Encoding target labels as integers...")
            self.y = self.y.astype('category').cat.codes

        print("Preprocessing complete.")

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        print("Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print("Data split complete.")

    def _create_model(self, model_type: str):
        """Return a model instance based on model_type."""
        if self.task_type != 'classification':
            raise ValueError("This comparison is focused on classification models.")

        if model_type == "random_forest":
            return RandomForestClassifier(
                random_state=42,
                n_estimators=300,
                n_jobs=-1
            )
        elif model_type == "decision_tree":
            return DecisionTreeClassifier(
                random_state=42,
                max_depth=None
            )
        elif model_type == "xgboost":
            return XGBClassifier(
                random_state=42,
                n_estimators=300,
                learning_rate=0.04,
                max_depth=4,
                n_jobs=-1,
                eval_metric="logloss"
            )
        else:
            raise ValueError("model_type must be 'random_forest', 'decision_tree', or 'xgboost'.")

    def _evaluate_classification(self, y_true, y_pred, y_proba=None):
        """Return dict of metrics for classification."""
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='binary')
        rec = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')

        roc = None
        # Binary ROC AUC if probabilities available
        if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 2:
            roc = roc_auc_score(y_true, y_proba[:, 1])

        print("\nClassification report:")
        print(classification_report(y_true, y_pred))

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        if roc is not None:
            print(f"ROC AUC  : {roc:.4f}")

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
        }

    def show_feature_importances(self, top_n: int | None = None):
        if not hasattr(self.model, "feature_importances_"):
            print("This model does not provide feature_importances_.")
            return

        # Get transformed feature names
        try:
            feature_names = self.preprocessor.get_feature_names_out()
        except AttributeError:
            num_names = self.numeric_features
            cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_names = cat_encoder.get_feature_names_out(self.categorical_features)
            feature_names = np.concatenate([num_names, cat_names])

        importances = self.model.feature_importances_

        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        if top_n is not None:
            fi_df = fi_df.head(top_n)

        print("\n=== Feature Importances ===")
        print(fi_df.to_string(index=False))

        return fi_df

    def train_and_evaluate_model(self, model_type: str):
        print("\n" + "=" * 50)
        print(f"Training model: {model_type}")
        print("=" * 50)

        self.model = self._create_model(model_type)
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)
        y_proba = None
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(self.X_test)

        metrics = self._evaluate_classification(self.y_test, y_pred, y_proba)
        self.show_feature_importances(top_n=15)
        metrics["model"] = model_type
        return metrics

    def run_all_models(self):
        # common preprocessing + split
        self.preprocess()
        self.split_data()

        model_types = ["decision_tree", "random_forest", "xgboost"]
        results = []

        for m in model_types:
            metrics = self.train_and_evaluate_model(m)
            results.append(metrics)

        # Summary table
        summary_df = pd.DataFrame(results)
        # move model column to front
        cols = ["model"] + [c for c in summary_df.columns if c != "model"]
        summary_df = summary_df[cols]

        print("\n" + "=" * 50)
        print("Overall comparison (higher is better):")
        print("=" * 50)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    processor = MLProcessor(
        file_path='Titanic-Dataset.csv',
        target_column='Survived'
    )
    processor.run_all_models()
