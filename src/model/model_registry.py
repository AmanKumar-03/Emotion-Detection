import os
import json
import logging

import mlflow
from mlflow import MlflowClient

# Logger
os.makedirs(
    "logs",
    exist_ok=True
)
logger = logging.getLogger("model_registry")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("model_registry_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =====================================================
# CI Protection
# =====================================================

if os.getenv("CI") == "true":

    logger.info(
        "CI environment detected. Skipping model registry."
    )

    exit(0)

# ==========================
# MLflow + DagsHub Configuration
# ==========================

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")


if DAGSHUB_USERNAME and DAGSHUB_TOKEN:

    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    logger.info("DagsHub authentication configured")

else:

    logger.warning(
        "DagsHub credentials missing. Running without authentication."
    )


mlflow.set_tracking_uri(
    "https://dagshub.com/AmanKumar-03/Emotion-Detection.mlflow"
)

mlflow.set_registry_uri(
    "https://dagshub.com/AmanKumar-03/Emotion-Detection.mlflow"
)


EXPERIMENT_NAME = "Emotion_Detection"

MODEL_NAME = "Emotion_Detection_Model1"


MIN_ACCURACY = 0.25
MIN_F1_SCORE = 0.25

# Load Metrics
def load_metrics():
    with open("./reports/metrics.json","r") as file:
        metrics = json.load(file)
    logger.info("Metrics loaded")
    return metrics

# Assign Champion Alias
def set_alias(
        model_name,
        alias,
        version
):

    try:

        client = MlflowClient()

        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version
        )

        logger.info(
            "Version %s assigned to %s",
            version,
            alias
        )


    except Exception as e:

        logger.error(
            "Alias update failed: %s",
            e
        )

        raise



def set_staging(model_name,version):

    set_alias(
        model_name,
        "staging",
        version
    )



def set_production(model_name,version):

    set_alias(
        model_name,
        "production",
        version
    )



def set_champion(model_name,version):

    set_alias(
        model_name,
        "champion",
        version
    )


# Register Model
def register_model():

    try:
        metrics = load_metrics()
        accuracy = metrics["accuracy"]
        f1 = metrics["f1_score"]
        logger.info("Accuracy %.4f",accuracy)
        logger.info("F1 %.4f",f1)

        if accuracy < MIN_ACCURACY:
            logger.warning("Accuracy below threshold")
            return

        if f1 < MIN_F1_SCORE:
            logger.warning("F1 below threshold")
            return

        client = MlflowClient()
        experiment = (
            client.get_experiment_by_name(
                EXPERIMENT_NAME
            )
        )

        if experiment is None:
            raise Exception("Experiment not found")

        runs = client.search_runs(
            experiment_ids=[
                experiment.experiment_id
            ],
            filter_string="attributes.status='FINISHED'",
            order_by=["metrics.test_accuracy DESC"],
            max_results=10
        )
        selected_run = None
        for run in runs:
            artifacts = client.list_artifacts(run.info.run_id)
            for artifact in artifacts:
                if artifact.path == "model":
                    selected_run = run
                    break
            if selected_run:
                break

        if selected_run is None:
            raise Exception("No model artifact found")
        run_id = (selected_run.info.run_id)
        model_uri = (f"runs:/{run_id}/model")
        logger.info("Registering model from %s",run_id)
        registered_model = (mlflow.register_model(model_uri,MODEL_NAME))
        version = registered_model.version
        logger.info("Model registered Version %s",version)

        # Set champion model
        set_staging(
            MODEL_NAME,
            version
        )
        if accuracy >= MIN_ACCURACY and f1 >= MIN_F1_SCORE:


            set_production(
                MODEL_NAME,
                version
            )


            set_champion(
                MODEL_NAME,
                version
            )
        print("\n==============================")
        print("MODEL REGISTRATION SUCCESS")
        print("Model:",MODEL_NAME)
        print("Version:",registered_model.version)
        print("Accuracy:",accuracy)
        print("F1 Score:",f1)
        print("==============================\n")
    except Exception as e:
        logger.exception("Model registration failed")
        raise

if __name__ == "__main__":
    register_model()