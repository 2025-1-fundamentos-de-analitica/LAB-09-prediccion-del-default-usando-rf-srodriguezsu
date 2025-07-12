import gzip
import json
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pickle


# Step 1: Data Cleaning
def clean_data(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df = df.dropna()
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x in [0, 1, 2, 3, 4] else 4)
    print(df.head())
    return df


# Step 2: Split Data
def split_data(df):
    x = df.drop(columns=["default"])
    y = df["default"]
    return train_test_split(x, y, test_size=0.2, random_state=42)


# Step 3: Create Pipeline
def create_pipeline():
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[("cat", categorical_transformer, categorical_features)]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )
    return pipeline


# Step 4: Optimize Hyperparameters
def optimize_pipeline(pipeline, x_train, y_train):
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
    }
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)

    return grid_search


# Step 5: Save Model
def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with gzip.open(file_path, "wb") as file:
        pickle.dump(model, file)


# Step 6: Calculate Metrics
def calculate_metrics(model, x, y, dataset_name):
    y_pred = model.predict(x)
    metrics = {
        "dataset": dataset_name,
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
    }
    return metrics


# Step 7: Calculate Confusion Matrix
def calculate_confusion_matrix(model, x, y, dataset_name):
    y_pred = model.predict(x)
    cm = confusion_matrix(y, y_pred)
    cm_dict = {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1]),
        },
    }
    return cm_dict


# Main Function
# Main Function
def main():
    train_data_path = "../files/input/train_data.csv.zip"
    test_data_path = "../files/input/test_data.csv.zip"
    model_path = "../files/models/model.pkl.gz"
    metrics_path = "../files/output/metrics.json"

    # Clean data
    train_df = clean_data(train_data_path)
    test_df = clean_data(test_data_path)

    # Split data
    x_train, x_test, y_train, y_test = split_data(train_df)

    # Create pipeline
    pipeline = create_pipeline()

    # Optimize pipeline
    best_model = optimize_pipeline(pipeline, x_train, y_train)

    # Save model
    save_model(best_model, model_path)

    # Calculate metrics
    train_metrics = calculate_metrics(best_model.best_estimator_, x_train, y_train, "train")
    test_metrics = calculate_metrics(best_model.best_estimator_, x_test, y_test, "test")

    # Calculate confusion matrix
    train_cm = calculate_confusion_matrix(best_model.best_estimator_, x_train, y_train, "train")
    test_cm = calculate_confusion_matrix(best_model.best_estimator_, x_test, y_test, "test")

    # Save metrics line by line
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as file:
        for metric in [train_metrics, test_metrics, train_cm, test_cm]:
            file.write(json.dumps(metric) + "\n")


if __name__ == "__main__":
    main()
