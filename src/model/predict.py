import os
import mlflow
import dagshub


# ==========================
# DagsHub MLflow Setup
# ==========================

dagshub.init(
    repo_owner="AmanKumar-03",
    repo_name="Emotion-Detection",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/AmanKumar-03/Emotion-Detection.mlflow"
)


# ==========================
# Load Production Model
# ==========================

MODEL_NAME = "Emotion_Detection_Model1"


model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}@production"
)


# ==========================
# Prediction
# ==========================

text = [
    "I am very happy today"
]


prediction = model.predict(text)


print("==============================")
print("Input:")
print(text[0])

print("------------------------------")

print("Prediction:")
print(prediction)

print("==============================")