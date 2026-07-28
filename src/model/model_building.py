import os
import pickle
import logging
import yaml
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier


logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)


if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_building_errors.log")
    file_handler.setLevel(logging.ERROR)
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
        logger.error("Parameter file not found: %s",e)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s",e)
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

def train_model(x_train: np.ndarray,y_train: np.ndarray,params: dict):
    """
    Train Gradient Boosting Classifier.
    This model supports multi-class classification.
    """
    try:
        model = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=42
            )
        model.fit(x_train,y_train)
        logger.info("Model training completed.")
        return model
    except Exception as e:
        logger.error("Training error: %s",e)
        raise

def save_model(model,file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file:
            pickle.dump(model,file)
        logger.info("Model saved at %s",file_path)
    except Exception as e:
        logger.error("Model saving error: %s",e)
        raise

def main():
    try:
        # Load parameters
        params = load_params("params.yaml")
        model_params = params["model_building"]

        # Load processed features
        train_data = load_data("./data/processed/train_bow.csv")
        test_data = load_data("./data/processed/test_bow.csv")

        # Split features and target
        X_train = train_data.drop("sentiment",axis=1).values
        y_train = train_data["sentiment"].values
        X_test = test_data.drop("sentiment",axis=1).values
        y_test = test_data["sentiment"].values
        logger.info("Training samples: %s",X_train.shape)

        # Train model
        model = train_model(X_train,y_train,model_params)

        # Save model
        save_model(model,"./models/model.pkl")
        logger.info("Model Building Completed Successfully.")
    except Exception as e:
        logger.error("Model pipeline failed: %s",e)
        print(f"Error: {e}")

if __name__ == "__main__":

    main()