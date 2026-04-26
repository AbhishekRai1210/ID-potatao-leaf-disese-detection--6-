import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

IMAGE_SIZE = 128

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "potato_disease_model.keras",
        compile=False   # 🔥 VERY IMPORTANT FIX
    )
    return model

@st.cache_data
def load_classes():
    with open("class_names.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
class_names = load_classes()

st.title("🌿 Potato Leaf Disease Detection")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        prediction = model.predict(img_array)
        index = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100

        st.success(f"Prediction: {class_names[index]}")
        st.write(f"Confidence: {confidence:.2f}%")