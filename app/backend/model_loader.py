import os
import pickle
import logging


# Logger configuration
logger = logging.getLogger("model_loader")
logger.setLevel(logging.INFO)

# Current backend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Artifacts folder path Docker container path: /app/artifacts
ARTIFACTS_DIR = os.path.join(BASE_DIR,"artifacts")

# Model files
MODEL_PATH = os.path.join(ARTIFACTS_DIR,"model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR,"vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR,"label_encoder.pkl")

def load_pickle(path, name):
    """Load pickle file safely"""
    try:
        with open(path, "rb") as file:
            obj = pickle.load(file)
        logger.info(f"{name} loaded successfully")
        return obj

    except FileNotFoundError:
        logger.error(f"{name} file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading {name}: {e}")
        raise

# Load Trained ML Model
model = load_pickle(MODEL_PATH,"Model")

# Load TF-IDF Vectorizer
vectorizer = load_pickle(VECTORIZER_PATH,"Vectorizer")

# Load Label Encoder
label_encoder = None
if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = load_pickle(LABEL_ENCODER_PATH,"Label Encoder")
else:
    logger.warning("Label encoder not found. Skipping...")