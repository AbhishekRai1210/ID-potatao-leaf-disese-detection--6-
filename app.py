import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

IMAGE_SIZE = 128

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("potato_disease_model.keras")

@st.cache_data
def load_class_names():
    with open("class_names.pkl", "rb") as f:
        return pickle.load(f)

model = load_trained_model()
class_names = load_class_names()

st.title("Potato Plant Disease Detection")
st.write("Upload a potato leaf image to detect the disease.")

uploaded_file = st.file_uploader(
    "Choose a potato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict Disease"):
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100

        st.success(f"Prediction: {class_names[predicted_index]}")
        st.write(f"Confidence: {confidence:.2f}%")
