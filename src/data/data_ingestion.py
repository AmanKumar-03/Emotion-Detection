import os
import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers if the script is executed multiple times
if not logger.handlers:

    # Display logs in terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Save only ERROR logs in a file
    file_handler = logging.FileHandler("error.log")
    file_handler.setLevel(logging.ERROR)

    # Log message format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    
def load_params(params_path: str) -> dict:

    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.info("Parameters loaded successfully.")
        return params
    except FileNotFoundError as e:
        logger.error("Parameter file not found: %s", e)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML Parsing Error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected Error: %s", e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.info("Dataset loaded successfully.")
        logger.info("Dataset Shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Error while loading dataset: %s", e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        # Remove ID column because it has no predictive value.
        df.drop(columns=["tweet_id"], inplace=True, errors="ignore")
        # Keep selected sentiments
        selected_sentiments = [
            "happiness",
            "sadness",
            "neutral",
            "love",
            "anger",
            "worry",
            "surprise",
            "hate",
            "enthusiasm",
            "fun",
            "relief"
        ]
        final_df = df[df["sentiment"].isin(selected_sentiments)].copy()
        # Label Encoding
        sentiment_mapping = {
            "happiness": 0,
            "sadness": 1,
            "neutral": 2,
            "love": 3,
            "anger": 4,
            "worry": 5,
            "surprise": 6,
            "hate": 7,
            "enthusiasm": 8,
            "fun": 9,
            "relief": 10
            }
        
        final_df["sentiment"] = final_df["sentiment"].map(sentiment_mapping)

        logger.info("Preprocessing Completed")

        logger.info("Remaining Dataset Shape: %s",final_df.shape)
        logger.info("\nSentiment Distribution:\n%s",final_df["sentiment"].value_counts())
        return final_df
    except Exception as e:
        logger.error("Preprocessing Error: %s", e)
        raise

def split_data(df: pd.DataFrame,test_size: float):

    train_data, test_data = train_test_split(df,test_size=test_size,random_state=42,stratify=df["sentiment"])
    logger.info("Train Shape: %s", train_data.shape)
    logger.info("Test Shape: %s", test_data.shape)
    return train_data, test_data

def save_data(train_data: pd.DataFrame,test_data: pd.DataFrame,data_path: str):

    try:
        raw_path = os.path.join(data_path, "raw")
        os.makedirs(raw_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_path, "train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_path, "test.csv"),index=False)
        logger.info("Train & Test datasets saved successfully.")
    except Exception as e:
        logger.error("Saving Error: %s", e)
        raise

def main():
    try:
        # Step 1
        params = load_params("params.yaml")
        test_size = params["data_ingestion"]["test_size"]
        # Step 2
        df = load_data("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")
        # Step 3
        final_df = preprocess_data(df)
        # Step 4
        train_data, test_data = split_data(final_df, test_size)
        # Step 5
        save_data(train_data, test_data, "./data")
        logger.info("Data Ingestion Completed Successfully.")
    except Exception as e:
        logger.error("Pipeline Failed: %s", e)
        print(e)
if __name__ == "__main__":
    main()