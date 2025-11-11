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

    def train_model(self):
        print("Training model...")
        if self.task_type == 'classification':
            self.model = RandomForestClassifier(
                random_state=42,
                n_estimators=200,
                n_jobs=-1
            )
        elif self.task_type == 'regression':
            self.model = RandomForestRegressor(
                random_state=42,
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

        print("\n=== Feature Importances ===")
        print(fi_df.to_string(index=False))

        return fi_df

    def run_pipeline(self):
        self.preprocess()
        self.split_data()
        self.train_model()
        self.evaluate_model()
        self.show_feature_importances(top_n=30)  # show top 30 by default


if __name__ == "__main__":
    processor = MLProcessor(
        file_path='Titanic-Dataset.csv',
        target_column='Survived'
    )

    print("Columns in dataset:", processor.df.columns.tolist())
    print(processor.df.head())

    processor.run_pipeline()
