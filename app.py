import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

IMAGE_SIZE = 128

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="potato_disease_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_class_names():
    with open("class_names.pkl", "rb") as f:
        return pickle.load(f)

interpreter = load_tflite_model()
class_names = load_class_names()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict Disease"):
        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]["index"])[0]
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"Prediction: {class_names[predicted_index]}")
        st.write(f"Confidence: {confidence:.2f}%")