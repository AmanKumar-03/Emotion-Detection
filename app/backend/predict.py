import logging

from app.backend.model_loader import (
    model,
    vectorizer
)

# Logger
logger = logging.getLogger("prediction")
logger.setLevel(logging.INFO)

# Prediction Function
def predict_emotion(text: str):

    try:
        if not text or text.strip() == "":
            raise ValueError("Input text cannot be empty")
        logger.info("Input text: %s",text)

        # Text -> TF-IDF
        vector = vectorizer.transform([text])
        logger.info("TF-IDF transformation completed")

        # Prediction
        prediction = model.predict(vector)[0]
        logger.info("Raw model prediction: %s",prediction)

        # Confidence
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vector)[0]
            confidence = float(max(probabilities))

        # Response
        result = {
            "emotion": str(prediction),
            "confidence": (
                round(confidence * 100, 2)
                if confidence is not None
                else None
            )
        }
        logger.info("Final result: %s",result)
        return result

    except Exception as e:
        logger.exception("Prediction error: %s",e)
        raise e