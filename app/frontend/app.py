import streamlit as st
import requests
from datetime import datetime

# PAGE CONFIG
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown(
    """
<style>
.main {
    background-color:#f7f9fc;
}
.title {
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#2E86C1;
}
.subtitle {
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:25px;
}
.result-box {
    background:white;
    border-radius:15px;
    padding:20px;
    box-shadow:0px 2px 15px rgba(0,0,0,0.15);
}
</style>
""",
    unsafe_allow_html=True
)


# SIDEBAR

st.sidebar.title("😊 Emotion Detection")
st.sidebar.markdown("---")

api_url = st.sidebar.text_input(
    "FastAPI URL",
    value="http://127.0.0.1:8000/predict"
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
### Instructions
1. Enter text
2. Click Predict
3. View emotion
4. Check confidence
"""
)

# TITLE

st.markdown(
    "<div class='title'>😊 Emotion Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>FastAPI + Streamlit + TF-IDF + Logistic Regression</div>",
    unsafe_allow_html=True
)

# SESSION HISTORY
if "history" not in st.session_state:
    st.session_state.history = []

# INPUT TEXT
text = st.text_area(
    "Enter your text",
    height=200,
    placeholder="Example: I am very happy today"
)

col1, col2 = st.columns(2)
predict_btn = col1.button("🚀 Predict",use_container_width=True)

clear_btn = col2.button("🗑 Clear",use_container_width=True)

if clear_btn:
    st.session_state.history = []
    st.rerun()

# EMOTION ICONS
emotion_icons = {
    "happiness":"😊",
    "enthusiasm":"😄",
    "fun":"😁",
    "love":"❤️",
    "sadness":"😢",
    "anger":"😠",
    "hate":"😡",
    "worry":"😨",
    "surprise":"😲",
    "neutral":"😐",
    "boredom":"😑",
    "empty":"😶",
    "relief":"😌"
}

# PREDICTION
if predict_btn:
    if text.strip() == "":
        st.warning("Please enter text")
        st.stop()

    try:
        with st.spinner("Predicting emotion..."):

            response = requests.post(
                api_url,
                json={
                    "text":text
                },
                timeout=20
            )

        if response.status_code != 200:
            st.error("FastAPI error")
            st.write(response.text)
            st.stop()
        result = response.json()

        # Correct FastAPI response parsing
        prediction = result["prediction"]
        emotion = prediction["emotion"]
        confidence = prediction.get("confidence",0)

        # Convert 48.67 -> 0.4867
        confidence_value = confidence / 100
        icon = emotion_icons.get(
            emotion.lower(),
            "🙂"
        )

        # Display Result
        st.markdown("---")

        st.markdown(
            "<div class='result-box'>",
            unsafe_allow_html=True
        )
        c1,c2 = st.columns(2)
        c1.metric("Predicted Emotion",f"{icon} {emotion.title()}")
        c2.metric("Confidence",f"{confidence:.2f}%")
        st.progress(confidence_value)
        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )
        st.success("Prediction completed successfully")

        # Save history
        st.session_state.history.append(
            {
                "time":datetime.now().strftime("%H:%M:%S"),
                "text":text,
                "emotion":emotion,
                "confidence":confidence_value
            }
        )
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI server")
    except Exception as e:
        st.error(str(e))


# HISTORY
if st.session_state.history:
    st.markdown("---")

    st.subheader("Prediction History")

    for item in reversed(st.session_state.history):
        
        with st.expander(f"{item['time']} | {item['emotion']}"):

            st.write("**Input Text**")

            st.write(item["text"])

            st.write("**Emotion**")
            st.success(item["emotion"].title())
            st.write("**Confidence**")
            st.progress(item["confidence"])
            st.write(f"{item['confidence']*100:.2f}%")


st.markdown("---")

st.caption(
    "Built with ❤️ using Streamlit + FastAPI + Scikit-Learn + MLflow"
)