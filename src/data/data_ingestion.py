import os
import logging
import yaml

import pandas as pd

import mlflow
import dagshub

from sklearn.model_selection import train_test_split


# DagsHub + MLflow Setup
import os

if os.getenv("CI") != "true":

    dagshub.init(repo_owner="AmanKumar-03",repo_name="Emotion-Detection",mlflow=True)

    mlflow.set_tracking_uri("https://dagshub.com/AmanKumar-03/Emotion-Detection.mlflow")

    mlflow.set_experiment("Emotion_Detection")


# Logger Configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)


if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("data_ingestion_error.log")
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
def load_data(url):

    try:
        df = pd.read_csv(url)
        logger.info("Dataset loaded successfully")
        logger.info("Dataset shape: %s",df.shape)
        logger.info("Columns: %s",df.columns.tolist())
        return df

    except Exception as e:
        logger.error("Dataset loading failed: %s",e)
        raise

# Data Cleaning
def preprocess_data(df):

    try:
        df = df.copy()
        # Remove id column
        df.drop(columns=["tweet_id"],inplace=True,errors="ignore")

        # Check columns
        required_columns = ["content","sentiment"]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Remove missing values
        df.dropna(subset=["content","sentiment"],inplace=True)

        # Remove duplicate rows
        before = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = (before - len(df))

        # Clean labels
        df["sentiment"] = (df["sentiment"].astype(str).str.lower().str.strip())

        # Remove invalid labels
        df = df[df["sentiment"] != ""]
        logger.info("Duplicates removed: %s",duplicates_removed)
        logger.info("Final shape: %s",df.shape)
        logger.info("Emotion classes:\n%s",df["sentiment"].value_counts())
        return df
    
    except Exception as e:
        logger.error("Preprocessing failed: %s",e)
        raise

# Train Test Split
def split_data(df,test_size,random_state):

    try:
        train_df, test_df = train_test_split(
            df,test_size=test_size,
            random_state=random_state,
            stratify=df["sentiment"])
        logger.info("Train shape: %s",train_df.shape)
        logger.info("Test shape: %s",test_df.shape)
        return train_df, test_df

    except Exception as e:
        logger.error("Split failed: %s",e)
        raise

# Save Data
def save_data(train_df,test_df):
    try:
        os.makedirs("./data/raw",exist_ok=True)
        train_df.to_csv("./data/raw/train.csv",index=False)
        test_df.to_csv("./data/raw/test.csv",index=False)
        logger.info("Raw data saved successfully")
    except Exception as e:
        logger.error("Saving failed: %s",e)
        raise

# Pipeline
def run_pipeline():
    params = load_params("params.yaml")
    data_params = params["data_ingestion"]
    test_size = data_params["test_size"]
    random_state = data_params["random_state"]

    if os.getenv("CI") != "true":
        mlflow.log_params(
            {
                "test_size": test_size,
                "random_state": random_state
            }
        )

    data_url = (
        "https://raw.githubusercontent.com/"
        "campusx-official/"
        "jupyter-masterclass/main/"
        "tweet_emotions.csv"
    )
    df = load_data(data_url)

    if os.getenv("CI") != "true":
        mlflow.log_param("dataset","tweet_emotions.csv")
        mlflow.log_metric("original_rows",len(df))

    processed_df = preprocess_data(df)

    if os.getenv("CI") != "true":
        mlflow.log_metric("processed_rows",len(processed_df))

    train_df, test_df = split_data(
        processed_df,
        test_size,
        random_state
    )

    if os.getenv("CI") != "true":
        mlflow.log_metric("train_rows",len(train_df))
        mlflow.log_metric("test_rows",len(test_df))
    save_data(train_df,test_df)
    logger.info("Data ingestion completed successfully")

# Main
def main():
    try:
        if os.getenv("CI") != "true":
            with mlflow.start_run(run_name="data_ingestion"):
                run_pipeline()
        else:
            run_pipeline()

    except Exception as e:
        logger.exception("Pipeline failed: %s",e)
        raise

if __name__ == "__main__":

    main()