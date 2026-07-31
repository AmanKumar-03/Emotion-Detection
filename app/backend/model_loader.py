import os
import pickle
import logging


logger = logging.getLogger("model_loader")
logger.setLevel(logging.INFO)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "vectorizer.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "label_encoder.pkl"
)

# Load Model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)
logger.info("Model loaded successfully")

# Load Vectorizer
with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)
logger.info("Vectorizer loaded successfully")

# Load Label Encoder
label_encoder = None

if os.path.exists(LABEL_ENCODER_PATH):
    with open(LABEL_ENCODER_PATH,"rb") as file:
        label_encoder = pickle.load(file)
    logger.info("Label encoder loaded")
else:
    logger.warning("Label encoder not found")