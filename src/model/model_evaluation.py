import os
import pickle
import json
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)


if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_evaluation_errors.log")
    file_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_model(file_path: str):

    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        logger.info("Model loaded from %s",file_path)
        return model
    except FileNotFoundError as e:
        logger.error("Model file not found: %s",e)
        raise
    except Exception as e:
        logger.error("Model loading error: %s",e)
        raise

def load_data(file_path: str):
    try:
        df = pd.read_csv(file_path)
        logger.info("Data loaded from %s",file_path)
        logger.info("Dataset Shape: %s",df.shape)
        return df
    
    except Exception as e:
        logger.error("Data loading error: %s",e)
        raise

def evaluate_model(model,X_test,y_test):
    try:
        # Prediction
        y_pred = model.predict(X_test)

        # Probability prediction
        y_pred_probability = model.predict_proba(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,average="weighted",zero_division=0)
        recall = recall_score(y_test,y_pred,average="weighted",zero_division=0)
        f1 = f1_score(y_test,y_pred,average="weighted",zero_division=0)
        roc_auc = roc_auc_score(y_test,y_pred_probability,multi_class="ovr")
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
            }
        logger.info("Evaluation completed successfully.")
        return metrics
    except Exception as e:
        logger.error("Evaluation error: %s",e)
        raise

def save_metrics(metrics: dict,file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            json.dump(metrics,file,indent=4)
        logger.info("Metrics saved at %s",file_path)
    except Exception as e:
        logger.error("Saving metrics error: %s",e)
        raise

def main():
    try:
        # Load model
        model = load_model("./models/model.pkl")

        # Load test data
        test_data = load_data("./data/processed/test_tfidf.csv")

        # Split features and target
        X_test = test_data.drop("sentiment",axis=1).values
        y_test = test_data["sentiment"].values
        logger.info("Testing data prepared.")

        # Evaluate
        metrics = evaluate_model(model,X_test,y_test)

        # Save results
        save_metrics(metrics,"./reports/metrics.json")
        logger.info("Model Evaluation Completed Successfully.")
    except Exception as e:
        logger.error("Evaluation pipeline failed: %s",e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()