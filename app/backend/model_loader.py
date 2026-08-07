import os
import pickle
import logging


logger = logging.getLogger("model_loader")
logger.setLevel(logging.INFO)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


ARTIFACTS_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)


MODEL_PATH = os.path.join(
    ARTIFACTS_DIR,
    "model.pkl"
)

VECTORIZER_PATH = os.path.join(
    ARTIFACTS_DIR,
    "vectorizer.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    ARTIFACTS_DIR,
    "label_encoder.pkl"
)



def load_pickle(path, name):

    try:
        with open(path, "rb") as file:
            obj = pickle.load(file)

        logger.info(f"{name} loaded successfully")

        return obj

    except FileNotFoundError:
        logger.error(f"{name} not found: {path}")
        raise



# Load Model
model = load_pickle(
    MODEL_PATH,
    "Model"
)


# Load Vectorizer
vectorizer = load_pickle(
    VECTORIZER_PATH,
    "Vectorizer"
)


# Load Label Encoder

label_encoder = None


if os.path.exists(LABEL_ENCODER_PATH):

    label_encoder = load_pickle(
        LABEL_ENCODER_PATH,
        "Label Encoder"
    )

else:

    logger.warning(
        "Label encoder not found. Skipping..."
    )