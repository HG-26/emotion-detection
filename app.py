import warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow protobuf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import av
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2

# Load your trained emotion model
model = load_model("emotion_model.h5")

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# MTCNN face detector
detector = MTCNN()

# Streamlit Page Config
st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("🎭 Emotion Detection App")

# --- Helper function to detect emotion from image ---
def detect_emotion_from_image(img_rgb):
    faces = detector.detect_faces(img_rgb)
    results = []
    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_img = img_rgb[y:y+h, x:x+w]
        if face_img.size != 0:
            face_pil = Image.fromarray(face_img).resize((48, 48)).convert("L")
            face_array = np.expand_dims(np.expand_dims(np.array(face_pil) / 255.0, axis=-1), axis=0)
            prediction = model.predict(face_array)
            emotion = EMOTIONS[np.argmax(prediction)]
            results.append((x, y, w, h, emotion))
    return results

# --- Live Webcam Transformer ---
class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = img[:, :, ::-1]

        results = detect_emotion_from_image(img_rgb)
        for (x, y, w, h, emotion) in results:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36, 255, 12), 2)
        return img

# --- Mode Selection ---
mode = st.radio("Choose Mode", ["📷 Live Webcam", "🖼️ Upload Image"])

if mode == "📷 Live Webcam":
    st.markdown("**Live Emotion Detection from Webcam**")
    webrtc_streamer(key="emotion", video_transformer_factory=EmotionTransformer)

elif mode == "🖼️ Upload Image":
    st.markdown("**Upload an Image to Detect Emotion**")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_rgb = np.array(img)
        results = detect_emotion_from_image(img_rgb)

        # Draw results
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        for (x, y, w, h, emotion) in results:
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_bgr, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36, 255, 12), 2)

        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Detected Emotions")
