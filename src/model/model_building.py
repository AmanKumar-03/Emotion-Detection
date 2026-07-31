import os
import pickle
import logging
import yaml
import time

import pandas as pd

import mlflow
import mlflow.sklearn

import dagshub

from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# DagsHub + MLflow Configuration
import os

if os.getenv("CI") != "true":
    dagshub.init(repo_owner="AmanKumar-03",repo_name="Emotion-Detection",mlflow=True)

    mlflow.set_tracking_uri("https://dagshub.com/AmanKumar-03/Emotion-Detection.mlflow")
    mlflow.set_experiment("Emotion_Detection")

# Logger Configuration
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("model_building_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Load Parameters
def load_params(path):

    try:
        with open(path,"r") as file:
            params = yaml.safe_load(file)
        logger.info("Parameters loaded successfully")
        return params

    except Exception as e:
        logger.error("Parameter loading failed: %s",e)
        raise

# Load Labels
def load_labels(path):

    try:
        df = pd.read_csv(path)
        labels = df["sentiment"].values
        logger.info("Labels loaded: %s",labels.shape)
        return labels

    except Exception as e:
        logger.error("Label loading failed: %s",e)
        raise

# Train Model
def train_model(X_train,y_train,model_params):

    try:

        model_type = model_params["model_type"]
        if model_type == "logistic_regression":
            model = LogisticRegression(
                C=model_params["C"],
                solver=model_params["solver"],
                max_iter=model_params["max_iter"],
                random_state=model_params["random_state"],
                class_weight="balanced"
            )
        else:
            raise ValueError(f"Unsupported model: {model_type}")

        logger.info("Training started...")
        start_time = time.time()
        model.fit(X_train,y_train)
        training_time = (time.time() - start_time)
        logger.info("Training completed in %.4f seconds",training_time)
        return (model,training_time)
    
    except Exception as e:
        logger.error("Training failed: %s",e)
        raise

# Save Model For FastAPI
def save_model(model):
    try:
        os.makedirs("artifacts",exist_ok=True)
        model_path = ("artifacts/model.pkl")
        with open(model_path,"wb") as file:
            pickle.dump(model,file)
        logger.info("Model saved: %s",model_path)
        return model_path

    except Exception as e:
        logger.error("Model saving failed: %s",e)
        raise

# Pipeline
def run_pipeline():
    params = load_params("params.yaml")
    model_params = params["model_building"]

    X_train = load_npz("./data/processed/train_tfidf.npz")
    X_test = load_npz("./data/processed/test_tfidf.npz")
    y_train = load_labels("./data/interim/train_processed.csv")
    y_test = load_labels("./data/interim/test_processed.csv")


    if X_train.shape[0] != len(y_train):
        raise ValueError("Training data mismatch")

    if X_test.shape[0] != len(y_test):
        raise ValueError("Testing data mismatch")

    model, training_time = train_model(
        X_train,
        y_train,
        model_params
    )
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {
        "train_accuracy":float(accuracy_score(y_train,train_pred)),
        "test_accuracy":float(accuracy_score(y_test,test_pred)),
        "precision":float(precision_score(y_test,test_pred,average="weighted",zero_division=0)),
        "recall":float(recall_score(y_test,test_pred,average="weighted",zero_division=0)),
        "f1_score":float(f1_score(y_test,test_pred,average="weighted",zero_division=0))
    }
    logger.info(metrics)

    # Only MLflow outside CI
    if os.getenv("CI") != "true":
        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time",training_time)
        mlflow.set_tag("model_type","Logistic Regression")
    model_path = save_model(model)

    if os.getenv("CI") != "true":
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="Emotion_Detection_Model1"
            )
    logger.info("Model building completed successfully")

# Main
def main():

    try:
        if os.getenv("CI") != "true":
            with mlflow.start_run(run_name="logistic_regression_best_model"):
                run_pipeline()
        else:
            run_pipeline()

    except Exception as e:
        logger.exception("Model building failed %s",e)
        raise

if __name__ == "__main__":

    main()