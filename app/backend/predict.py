import logging

from model_loader import (
    model,
    vectorizer,
    label_encoder
)

logger = logging.getLogger("prediction")
logger.setLevel(logging.INFO)

def predict_emotion(text):
    try:
        # Convert text into TF-IDF
        vector = vectorizer.transform([text])

        # Prediction
        prediction = model.predict(vector)[0]

        # Confidence
        confidence = None

        if hasattr(model,"predict_proba"):
            probabilities = model.predict_proba(vector)[0]
            confidence = float(max(probabilities))

        # Convert label back
        if label_encoder:
            prediction = label_encoder.inverse_transform([prediction])[0]

        result = {
            "emotion": prediction,
            "confidence": (
                round(confidence * 100,2)
                if confidence
                else None
            )
        }
        logger.info("Prediction result: %s",result)
        return result

    except Exception as e:
        logger.exception("Prediction failed")
        raise e