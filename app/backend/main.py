from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.backend.predict import predict_emotion
from app.backend.schema import TextInput



app = FastAPI(

    title="Emotion Detection API",
    description=(
        "Machine Learning API for detecting emotions "
        "from text using TF-IDF and Logistic Regression"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


app.add_middleware(

    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def home():

    return {
        "message": "Emotion Detection API is Running",
        "status": "Success",
        "version": "1.0.0"
    }


@app.get("/health",tags=["System"])
def health_check():

    return {
        "status": "Healthy",
        "service": "Emotion Detection API"
    }


@app.post("/predict",tags=["Emotion Prediction"])
def predict_emotion_api(data: TextInput):

    try:

        result = predict_emotion(data.text)
        return {
            "success": True,
            "input_text": data.text,
            "prediction": result
        }
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))


@app.get("/info")
def api_info():

    return {
        "project": "Emotion Detection",
        "framework": "FastAPI",
        "model": "Logistic Regression",
        "vectorizer": "TF-IDF",
        "tracking": "MLflow + DagsHub",
        "deployment": "FastAPI + Streamlit",
        "author": "Aman Kumar"
    }

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )