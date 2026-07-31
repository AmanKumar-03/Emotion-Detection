import streamlit as st
import requests
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#f7f9fc;
}

.title{
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#2E86C1;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:25px;
}

.result-box{
    background:#ffffff;
    border-radius:15px;
    padding:20px;
    box-shadow:0px 2px 15px rgba(0,0,0,0.15);
}

textarea{
    font-size:18px !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
    width=120
)

st.sidebar.title("Emotion Detection")

st.sidebar.markdown("---")

api_url = st.sidebar.text_input(
    "FastAPI URL",
    value="http://127.0.0.1:8000/predict"
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
### Instructions

1. Enter any sentence.

2. Click Predict.

3. View emotion.

4. Check confidence score.
"""
)

# -----------------------------
# TITLE
# -----------------------------
st.markdown(
    "<div class='title'>😊 Emotion Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>FastAPI + Streamlit + Machine Learning</div>",
    unsafe_allow_html=True
)

# -----------------------------
# SESSION STATE
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# INPUT
# -----------------------------
text = st.text_area(
    "Enter your text",
    height=200,
    placeholder="Example: I am extremely happy today!"
)

col1, col2 = st.columns(2)

predict = col1.button("🚀 Predict", use_container_width=True)

clear = col2.button("🗑 Clear", use_container_width=True)

if clear:
    st.rerun()

# -----------------------------
# PREDICT
# -----------------------------
if predict:

    if text.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Predicting emotion..."):

        try:

            response = requests.post(
                api_url,
                json={"text": text},
                timeout=20
            )

            if response.status_code != 200:
                st.error("FastAPI returned an error.")
                st.stop()

            result = response.json()

            emotion = result["emotion"]
            confidence = result.get("confidence", 0)

            emoji = {
                "joy":"😊",
                "happy":"😊",
                "sadness":"😢",
                "sad":"😢",
                "anger":"😠",
                "fear":"😨",
                "love":"❤️",
                "surprise":"😲",
                "neutral":"😐"
            }

            icon = emoji.get(emotion.lower(),"🙂")

            st.markdown("---")

            st.markdown(
                "<div class='result-box'>",
                unsafe_allow_html=True
            )

            c1,c2 = st.columns(2)

            c1.metric(
                label="Predicted Emotion",
                value=f"{icon} {emotion.title()}"
            )

            c2.metric(
                label="Confidence",
                value=f"{confidence:.2%}"
            )

            st.progress(float(confidence))

            st.markdown("</div>", unsafe_allow_html=True)

            st.success("Prediction completed successfully.")

            st.session_state.history.append(
                {
                    "time":datetime.now().strftime("%H:%M:%S"),
                    "text":text,
                    "emotion":emotion,
                    "confidence":confidence
                }
            )

        except requests.exceptions.ConnectionError:

            st.error("Cannot connect to FastAPI server.")

        except Exception as e:

            st.error(str(e))

# -----------------------------
# HISTORY
# -----------------------------
if len(st.session_state.history)>0:

    st.markdown("---")

    st.subheader("Prediction History")

    for item in reversed(st.session_state.history):

        with st.expander(f"{item['time']}  |  {item['emotion']}"):

            st.write("**Input Text**")

            st.write(item["text"])

            st.write("**Emotion**")

            st.success(item["emotion"])

            st.write("**Confidence**")

            st.progress(float(item["confidence"]))

            st.write(f"{item['confidence']:.2%}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.caption(
    "Built with ❤️ using Streamlit + FastAPI + Scikit-Learn"
)