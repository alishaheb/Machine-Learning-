import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from typing import Optional
import shap
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg"



RANDOMSTATE = 42
IMPORTANT = 7  # Number of top features to display
#dataframe=pd.DataFrame('')
class MLProcessor:
    def __init__(self, file_path: str, target_column: str):
        self.file_path = file_path
        self.target_column = target_column
        self.df = pd.read_csv(file_path)

        self.X = None
        self.y = None
        self.model = None
        self.task_type = None

        self.preprocessor: Optional[ColumnTransformer] = None
        self.numeric_features = None
        self.categorical_features = None

    def preprocess(self):
        print("Starting preprocessing...")
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(axis=1, how='all', inplace=True)

        # Drop irrelevant or unique identifier columns (Titanic-specific)
        drop_cols = [col for col in ['PassengerId', 'Name', 'Ticket', 'Cabin']
                     if col in self.df.columns]
        if drop_cols:
            print(f"Dropping columns: {drop_cols}")
            self.df.drop(columns=drop_cols, inplace=True)

        # Separate target and features
        self.y = self.df[self.target_column]
        self.X = self.df.drop(columns=[self.target_column])

        # Determine task type
        if ((self.y.nunique() <= 20 and self.y.dtype == 'object')
                or self.y.dtype == 'bool'):
            self.task_type = 'classification'
        elif self.y.nunique() <= 2:
            self.task_type = 'classification'
        elif self.y.dtype in ['int64', 'float64'] and self.y.nunique() > 20:
            self.task_type = 'regression'
        else:
            raise ValueError("Unable to determine task type. "
                             "Please check the target column.")

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
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
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

    def split_data(self, test_size: float = 0.2, random_state: int = RANDOMSTATE):
        print("Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print("Data split complete..")

    def train_model(self):
        print("Training model...")
        if self.task_type == 'classification':
            self.model = RandomForestClassifier(
                random_state=RANDOMSTATE,
                n_estimators=200,
                n_jobs=-1
            )
        elif self.task_type == 'regression':
            self.model = RandomForestRegressor(
                random_state=RANDOMSTATE,
                n_estimators=200,
                n_jobs=-1
            )
        else:
            raise ValueError("Task type must be 'classification' or 'regression'.")

        self.model.fit(self.X_train, self.y_train)
        print("Model training complete.")

    def evaluate_model(self):
        print("Evaluating model...")
        y_pred = self.model.predict(self.X_test)
        if self.task_type == 'classification':
            print(classification_report(self.y_test, y_pred))
        elif self.task_type == 'regression':
            mse = mean_squared_error(self.y_test, y_pred)
            print(f"Mean Squared Error: {mse:.2f}")

    def show_feature_importances(self, top_n: Optional[int] = None):
        """
        Print feature importances sorted from highest to lowest.
        Works for both classification and regression RandomForest.
        """
        if not hasattr(self.model, "feature_importances_"):
            print("This model does not provide feature_importances_.")
            return

        # Get transformed feature names
        try:
            feature_names = self.preprocessor.get_feature_names_out()
        except AttributeError:
            # Fallback for older sklearn
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

        print("\n== Feature Importances ==")
        print(fi_df.to_string(index=False))

        return fi_df

    def explain_with_shap(self, max_background: int = 200, max_explain: int = 200):
        """
        SHAP explanation for the trained RandomForest model.
        Uses TreeExplainer and shows:
          - summary bar plot (global importance)
          - summary beeswarm plot (direction + spread)
        """

        if self.model is None:
            raise ValueError("Model is not trained yet. Call train_model() first.")

        print("Running SHAP...")

        # SHAP is faster if we sample
        X_train_sample = self.X_train
        X_test_sample = self.X_test

        if hasattr(X_train_sample, "shape") and X_train_sample.shape[0] > max_background:
            idx = np.random.RandomState(RANDOMSTATE).choice(X_train_sample.shape[0], max_background, replace=False)
            X_train_sample = X_train_sample[idx]

        if hasattr(X_test_sample, "shape") and X_test_sample.shape[0] > max_explain:
            idx = np.random.RandomState(RANDOMSTATE).choice(X_test_sample.shape[0], max_explain, replace=False)
            X_test_sample = X_test_sample[idx]

        # Convert sparse matrices (common after OneHotEncoder) to dense for plotting compatibility
        if hasattr(X_train_sample, "toarray"):
            X_train_sample_dense = X_train_sample.toarray()
        else:
            X_train_sample_dense = X_train_sample

        if hasattr(X_test_sample, "toarray"):
            X_test_sample_dense = X_test_sample.toarray()
        else:
            X_test_sample_dense = X_test_sample

        # Feature names from the preprocessor
        try:
            feature_names = self.preprocessor.get_feature_names_out()
        except AttributeError:
            num_names = self.numeric_features
            cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_names = cat_encoder.get_feature_names_out(self.categorical_features)
            feature_names = np.concatenate([num_names, cat_names])

        # TreeExplainer is the right choice for RandomForest
        explainer = shap.TreeExplainer(self.model)

        shap_values = explainer.shap_values(X_test_sample_dense)

        # --- Plotting ---
        # For binary classification, shap_values is usually a list [class0, class1]
        if self.task_type == "classification" and isinstance(shap_values, list):
            # Explain the positive class (class 1) by convention
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        # Global importance (bar)
        shap.summary_plot(sv, X_test_sample_dense, feature_names=feature_names, plot_type="bar")

        # Global importance with direction (beeswarm)
        shap.summary_plot(sv, X_test_sample_dense, feature_names=feature_names)

        print("SHAP complete.")

    def run_pipeline(self):
        self.preprocess()
        self.split_data()
        self.train_model()
        self.evaluate_model()
        self.show_feature_importances(top_n=IMPORTANT)  # show top 30 by default
        self.explain_with_shap()

if __name__ == "__main__":
    processor = MLProcessor(
        file_path='datasets/alzheimers_disease_data.csv',
        target_column='Diagnosis'
    )

    print("Columns in dataset:", processor.df.columns.tolist())
    print(processor.df.head())

    processor.run_pipeline()
