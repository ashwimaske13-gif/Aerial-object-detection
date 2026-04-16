import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# MODEL_PATH = "bird_drone_model.h5"
MODEL_PATH = "bird_drone_model.keras"

st.set_page_config(page_title="Bird vs Drone Detector", page_icon="✈️")

st.title("🛰️ Aerial Object Classification")
st.write("Upload an image to classify it as Bird or Drone")

# Load model
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model(MODEL_PATH)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success(f"🚁 Drone Detected ({prediction:.2f})")
    else:
        st.success(f"🐦 Bird Detected ({1 - prediction:.2f})")