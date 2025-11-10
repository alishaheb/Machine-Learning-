from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, Imputer
)
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import when

spark = SparkSession.builder.appName("HP_Treatment").getOrCreate()

# 1. Load data
df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("Hp treatment training.csv")
)

# 2. Drop useless columns
cols_to_drop = [
    "ID_Patient_Care_Situation",
    "Patient_ID",
    "Unnamed: 18", "Unnamed: 19", "Unnamed: 20", "Unnamed: 21"
]
for c in cols_to_drop:
    if c in df.columns:
        df = df.drop(c)

# 3. Define columns
label_col = "Survived_1_year"

categorical_cols = [
    "Treated_with_drugs",
    "Patient_Smoker",
    "Patient_Rural_Urban",
    "Patient_mental_condition",
]

numeric_cols = [
    "Diagnosed_Condition",
    "Patient_Age",
    "Patient_Body_Mass_Index",
    "A", "B", "C", "D", "E", "F", "Z",
    "Number_of_prev_cond",
]

# Ensure label is numeric double (if it’s int it’s usually fine too)
df = df.withColumn(label_col, col(label_col).cast("double"))

# 4. Train / test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 5. Impute numeric missing values (if any)
imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=[c + "_imp" for c in numeric_cols]
)

imputed_numeric_cols = [c + "_imp" for c in numeric_cols]

# 6. Index + OneHotEncode categorical
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_idx",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(
        inputCol=c + "_idx",
        outputCol=c + "_ohe"
    )
    for c in categorical_cols
]

ohe_cols = [c + "_ohe" for c in categorical_cols]

# 7. Assemble all features
assembler = VectorAssembler(
    inputCols=imputed_numeric_cols + ohe_cols,
    outputCol="features"
)

# 8. Define models
lr = LogisticRegression(
    featuresCol="features",
    labelCol=label_col,
    maxIter=100
)

dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol=label_col,
    maxDepth=5  # you can tune this
)

# 9. Pipelines
lr_pipeline = Pipeline(stages=[imputer] + indexers + encoders + [assembler, lr])
dt_pipeline = Pipeline(stages=[imputer] + indexers + encoders + [assembler, dt])

# 10. Fit models
lr_model = lr_pipeline.fit(train_df)
dt_model = dt_pipeline.fit(train_df)

# 11. Predictions
lr_pred = lr_model.transform(test_df)
dt_pred = dt_model.transform(test_df)

# 12. Evaluate – ROC AUC
binary_eval = BinaryClassificationEvaluator(
    labelCol=label_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

lr_auc = binary_eval.evaluate(lr_pred)
dt_auc = binary_eval.evaluate(dt_pred)

print("Logistic Regression ROC AUC:", lr_auc)
print("Decision Tree ROC AUC:", dt_auc)

# 13. Helper: compute confusion matrix + metrics manually
def compute_metrics(pred_df, label_col="Survived_1_year"):
    # Convert probabilities to predicted label if not already
    if "prediction" not in pred_df.columns:
        pred_df = pred_df.withColumn("prediction", when(col("probability")[1] >= 0.5, 1.0).otherwise(0.0))

    cm = (
        pred_df.groupBy(label_col, "prediction")
        .count()
        .collect()
    )

    # Initialize
    tp = fp = tn = fn = 0
    for row in cm:
        label = int(row[label_col])
        pred = int(row["prediction"])
        count = row["count"]
        if label == 1 and pred == 1:
            tp = count
        elif label == 0 and pred == 1:
            fp = count
        elif label == 0 and pred == 0:
            tn = count
        elif label == 1 and pred == 0:
            fn = count

    accuracy = (tp + tn) / float(tp + tn + fp + fn)
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / float(precision + recall) if (precision + recall) > 0 else 0.0

    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 score :", f1)

print("\n=== Logistic Regression Metrics ===")
compute_metrics(lr_pred, label_col)

print("\n=== Decision Tree Metrics ===")
compute_metrics(dt_pred, label_col)
