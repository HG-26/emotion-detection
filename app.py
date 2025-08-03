import streamlit as st
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load trained model
model = load_model("emotion_model.h5")

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# MTCNN face detector
detector = MTCNN()

st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("🎭 Real-Time Emotion Detection (No OpenCV)")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect faces
    image_np = np.array(image)
    faces = detector.detect_faces(image_np)

    if faces:
        for face in faces:
            x, y, w, h = face['box']
            face_img = image_np[y:y+h, x:x+w]
            face_img = Image.fromarray(face_img).resize((48, 48)).convert("L")
            face_array = np.expand_dims(np.expand_dims(np.array(face_img) / 255.0, axis=-1), axis=0)

            # Predict emotion
            prediction = model.predict(face_array)
            emotion = EMOTIONS[np.argmax(prediction)]

            st.write(f"Detected Emotion: **{emotion}**")
    else:
        st.warning("No faces detected in the image.")

