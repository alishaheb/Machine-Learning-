import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from typing import Optional, Union

class MLProcessor:
    def __init__(self, file_path: str, target_column: str):
        self.file_path = file_path
        self.target_column = target_column
        self.df = pd.read_csv(file_path)
        self.X = None
        self.y = None
        self.model = None
        self.task_type = None

    def preprocess(self):
        print("Starting preprocessing...")
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(axis=1, how='all', inplace=True)

        # Drop irrelevant or unique identifier columns
        drop_cols = [col for col in ['PassengerId', 'Name', 'Ticket', 'Cabin'] if col in self.df.columns]
        self.df.drop(columns=drop_cols, inplace=True)

        self.y = self.df[self.target_column]
        self.X = self.df.drop(columns=[self.target_column])

        # Determine task type
        if self.y.nunique() <= 20 and self.y.dtype == 'object' or self.y.dtype == 'bool':
            self.task_type = 'classification'
        elif self.y.dtype in ['int64', 'float64'] and self.y.nunique() > 20:
            self.task_type = 'regression'
        elif self.y.nunique() <= 2:
            self.task_type = 'classification'
        else:
            raise ValueError("Unable to determine task type. Please check the target column.")

        # Identify feature types
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Pipelines
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

        self.X = preprocessor.fit_transform(self.X)

        if self.task_type == 'classification' and self.y.dtype != 'int':
            self.y = LabelEncoder().fit_transform(self.y)

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
            self.model = RandomForestClassifier(random_state=42)
        elif self.task_type == 'regression':
            self.model = RandomForestRegressor(random_state=42)
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

    def run_pipeline(self):
        self.preprocess()
        self.split_data()
        self.train_model()
        self.evaluate_model()




if __name__ == "__main__":
    processor = MLProcessor(file_path='Titanic-Dataset.csv', target_column='Survived')
    processor.df.head()
    print(processor.df.columns)
    processor.run_pipeline()