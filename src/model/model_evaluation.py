import os
import pickle
import logging
import json

import pandas as pd
import scipy.sparse as sp

import mlflow
import dagshub

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# DagsHub + MLflow Setup
dagshub.init(repo_owner="AmanKumar-03",repo_name="Emotion-Detection",mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/AmanKumar-03/Emotion-Detection.mlflow")

mlflow.set_experiment("Emotion_Detection")

# Logger
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)


if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("model_evaluation_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Load Features
def load_features(path):
    try:
        X = sp.load_npz(path)
        logger.info("Features loaded: %s",X.shape)
        return X

    except Exception as e:
        logger.error("Feature loading failed: %s",e)
        raise

# Load Labels
def load_labels(path):
    try:
        df = pd.read_csv(path)
        y = df["sentiment"].values
        logger.info("Labels loaded: %s",y.shape)
        return y

    except Exception as e:
        logger.error("Label loading failed: %s",e)
        raise

# Load Model
def load_model(path):
    try:
        with open(path,"rb") as file:
            model = pickle.load(file)
        logger.info("Model loaded successfully")
        return model

    except Exception as e:
        logger.error("Model loading failed: %s",e)
        raise

# Save Confusion Matrix
def save_confusion_matrix(cm):
    os.makedirs("./reports",exist_ok=True)
    plt.figure(figsize=(7,6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    path = ("./reports/confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    return path

# Save Metrics
def save_metrics(metrics):
    os.makedirs("./reports",exist_ok=True)
    path = ("./reports/metrics.json")
    with open(path,"w") as file:
        json.dump(metrics,file,indent=4)
    return path

# Main
def main():
    with mlflow.start_run(run_name="model_evaluation"):

        try:
            # Load test data
            X_test = load_features("./data/processed/test_tfidf.npz")
            y_test = load_labels("./data/interim/test_processed.csv")

            # IMPORTANT:
            # Model from artifacts folder
            model = load_model("./artifacts/model.pkl")

            # Prediction
            predictions = model.predict(X_test)

            # Metrics
            metrics = {
                "accuracy":float(accuracy_score(y_test,predictions)),
                "precision":float(precision_score(y_test,predictions,average="weighted",zero_division=0)),
                "recall":float(recall_score(y_test,predictions,average="weighted",zero_division=0)),
                "f1_score":float(f1_score(y_test,predictions,average="weighted",zero_division=0))
            }
            logger.info(metrics)

            # Classification report
            report = classification_report(y_test,predictions,zero_division=0)
            print(report)
            os.makedirs("./reports",exist_ok=True)
            report_path = ("./reports/classification_report.txt")
            with open(report_path,"w") as file:
                file.write(report)

            # Confusion Matrix
            cm = confusion_matrix(y_test,predictions)
            cm_path = save_confusion_matrix(cm)
            metrics_path = save_metrics(metrics)

            # MLflow logging
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(metrics_path)
            mlflow.log_artifact(report_path)
            logger.info("Model evaluation completed successfully")
        except Exception as e:
            logger.exception("Evaluation pipeline failed: %s",e)
            raise

if __name__ == "__main__":
    main()