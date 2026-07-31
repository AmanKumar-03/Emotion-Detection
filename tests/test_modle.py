import os
import pickle


MODEL_PATH = "artifacts/model.pkl"
VECTORIZER_PATH = "artifacts/vectorizer.pkl"


def test_model_files_exist():

    assert os.path.exists(
        MODEL_PATH
    ), "Model file missing"


    assert os.path.exists(
        VECTORIZER_PATH
    ), "Vectorizer file missing"



def test_model_prediction():

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        model = pickle.load(file)


    with open(
        VECTORIZER_PATH,
        "rb"
    ) as file:

        vectorizer = pickle.load(file)


    text = [
        "I am very happy today"
    ]


    vector = vectorizer.transform(
        text
    )


    prediction = model.predict(
        vector
    )


    assert prediction is not None

    assert len(prediction) == 1