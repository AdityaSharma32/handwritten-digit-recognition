import os
import numpy as np
import streamlit as st
import tensorflow as tf
from src.preprocessing import prepare_digit_image

MODEL_PATH = "model/digit_cnn.keras"

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("AI-Based Handwritten Digit Recognition")
st.write(
    "Upload a clear image containing one handwritten digit (0–9). "
    "The CNN will predict the digit and show its confidence score."
)

if not os.path.exists(MODEL_PATH):
    st.error("The trained model was not found. First run: python train.py")
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()
uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
    try:
        original_preview = uploaded_file.getvalue()
        st.image(original_preview, caption="Uploaded image", width=220)

        processed_image, image_array = prepare_digit_image(uploaded_file)
        probabilities = model.predict(image_array, verbose=0)[0]
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit]) * 100

        left, right = st.columns(2)
        with left:
            st.subheader("Prediction")
            st.metric("Recognized digit", predicted_digit)
        with right:
            st.subheader("Confidence")
            st.metric("Model confidence", f"{confidence:.2f}%")

        st.image(processed_image, caption="Preprocessed 28 × 28 image", width=140)
        st.write("Scores for each digit")
        st.bar_chart({str(i): float(probabilities[i]) for i in range(10)})

        st.caption(
            "Confidence is the model's estimated probability, not a guarantee "
            "that the prediction is correct."
        )
    except Exception as error:
        st.error(f"Could not process this image: {error}")
