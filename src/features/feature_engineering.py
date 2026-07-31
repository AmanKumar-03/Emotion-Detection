import os
import logging
import yaml
import pickle

import numpy as np
import pandas as pd

import mlflow
import dagshub

from scipy.sparse import save_npz

from sklearn.feature_extraction.text import TfidfVectorizer


# DagsHub + MLflow Configuration
dagshub.init(repo_owner="AmanKumar-03",repo_name="Emotion-Detection",mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/AmanKumar-03/Emotion-Detection.mlflow")

mlflow.set_experiment("Emotion_Detection")

# Logger Configuration
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("feature_engineering_error.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Load Parameters
def load_params(path):
    try:
        with open(path, "r") as file:
            params = yaml.safe_load(file)
        logger.info("Parameters loaded successfully")
        return params

    except Exception as e:
        logger.error("Parameter loading failed: %s",e)
        raise

# Load Dataset
def load_data(path):

    try:
        df = pd.read_csv(path)
        required_columns = ["content","sentiment"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        logger.info("Loaded dataset: %s",path)
        logger.info("Dataset shape: %s",df.shape)
        return df

    except Exception as e:
        logger.error("Data loading failed: %s",e)
        raise

# TF-IDF Feature Engineering
def apply_tfidf(train_data,test_data,feature_params):

    vectorizer = TfidfVectorizer(
        max_features=
        feature_params["max_features"],
        ngram_range=(
            feature_params["ngram_min"],
            feature_params["ngram_max"]
        ),
        min_df=feature_params["min_df"],
        max_df=feature_params["max_df"])
    
    X_train_text = train_data["content"]
    X_test_text = test_data["content"]
    y_train = train_data["sentiment"]
    y_test = test_data["sentiment"]
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    logger.info("TF-IDF transformation completed")
    logger.info("Training features shape: %s",X_train.shape)
    logger.info("Testing features shape: %s",X_test.shape)
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        vectorizer
    )

# Save Vectorizer
def save_vectorizer(vectorizer):
    try:
        os.makedirs("artifacts",exist_ok=True)
        path = ("artifacts/vectorizer.pkl")
        with open(path,"wb") as file:
            pickle.dump(vectorizer,file)
        logger.info("Vectorizer saved: %s",path)
        return path

    except Exception as e:
        logger.error("Vectorizer saving failed: %s",e)
        raise

# Main Pipeline
def main():
    with mlflow.start_run(run_name="tfidf_feature_engineering"):

        try:
            params = load_params("params.yaml")
            feature_params = params["feature_engineering"]

            # Log parameters
            mlflow.log_params({
                    "vectorizer":feature_params["vectorizer"],
                    "max_features":feature_params["max_features"],
                    "ngram_min":feature_params["ngram_min"],
                    "ngram_max":feature_params["ngram_max"],
                    "min_df":feature_params["min_df"],
                    "max_df":feature_params["max_df"]

                })
            train_data = load_data("./data/interim/train_processed.csv")
            test_data = load_data("./data/interim/test_processed.csv")
            if feature_params["vectorizer"] != "tfidf":
                raise ValueError("Only TF-IDF vectorizer supported")

            (X_train,X_test,y_train,y_test,vectorizer) = apply_tfidf(
                train_data,test_data,feature_params)

            # Save processed directory
            os.makedirs("./data/processed",exist_ok=True)

            # Save features
            save_npz("./data/processed/train_tfidf.npz",X_train)
            save_npz("./data/processed/test_tfidf.npz",X_test)

            # Save labels
            np.save("./data/processed/train_labels.npy",np.array(y_train))
            np.save("./data/processed/test_labels.npy",np.array(y_test))

            # Save vectorizer for FastAPI
            vectorizer_path = save_vectorizer(vectorizer)

            # MLflow logging
            mlflow.log_params({
                    "vocabulary_size":len(vectorizer.vocabulary_),
                    "train_features":X_train.shape[1],
                    "test_features":X_test.shape[1]
                })
            mlflow.log_artifact(vectorizer_path)
            logger.info("Feature Engineering completed successfully")
        except Exception as e:
            logger.exception("Pipeline failed: %s",e)
            raise

if __name__ == "__main__":

    main()