import os
import logging
import yaml
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)


if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("feature_engineering_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)


    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """
    Load parameters from params.yaml
    """
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.info("Parameters loaded successfully.")
        return params
    except FileNotFoundError as e:
        logger.error("Parameter file not found: %s",e)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML Error: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s",e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load processed dataset
    """
    try:
        df = pd.read_csv(file_path)
        logger.info("Data loaded from %s",file_path)
        logger.info("Dataset Shape: %s",df.shape)
        return df
    
    except Exception as e:
        logger.error("Error loading data: %s",e)
        raise

def apply_tfidf(train_data: pd.DataFrame,test_data: pd.DataFrame,max_features: int):
    """
    Convert text data into numerical vectors using CountVectorizer.
    Train:
        fit_transform()
    Test:
        transform()
    Because test data should use
    the vocabulary learned from training data.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features,ngram_range=(1,2),min_df=2)

        # Text column
        X_train = train_data["content"]
        X_test = test_data["content"]

        # Target labels
        y_train = train_data["sentiment"]
        y_test = test_data["sentiment"]

        # Fit only on training data
        X_train_tfidf = vectorizer.fit_transform(X_train)

        # Transform test data
        X_test_tfidf = vectorizer.transform(X_test)
        logger.info("Vocabulary Size: %s",len(vectorizer.vocabulary_))

        # Convert sparse matrix to dataframe
        train_features = pd.DataFrame(X_train_tfidf.toarray(),columns=vectorizer.get_feature_names_out())
        test_features = pd.DataFrame(X_test_tfidf.toarray(),columns=vectorizer.get_feature_names_out())

        # Add target column
        train_features["sentiment"] = y_train.values
        test_features["sentiment"] = y_test.values
        logger.info("Bag of Words transformation completed.")
        logger.info("Train Feature Shape: %s",train_features.shape)
        logger.info("Test Feature Shape: %s",test_features.shape)
        return train_features, test_features
    except Exception as e:
        logger.error("Feature engineering failed: %s",e)
        raise

def save_data(df: pd.DataFrame,file_path: str):
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory,exist_ok=True)
        df.to_csv( file_path,index=False)
        logger.info("File saved successfully: %s",file_path)
    except Exception as e:
        logger.error("Error saving file: %s",e)
        raise

def main():
    try:
        # Load parameters
        params = load_params("params.yaml")
        max_features = params["feature_engineering"]["max_features"]

        # Load processed data
        train_data = load_data("./data/interim/train_processed.csv")
        test_data = load_data("./data/interim/test_processed.csv")

        # Apply BOW
        train_df, test_df = apply_tfidf(train_data,test_data,max_features)
        # Save features
        save_data(train_df,"./data/processed/train_bow.csv")
        save_data(test_df,"./data/processed/test_bow.csv")
        logger.info("Feature Engineering Completed Successfully.")
    except Exception as e:
        logger.error("Pipeline failed: %s",e)
        print(f"Error: {e}")

if __name__ == "__main__":
    
    main()