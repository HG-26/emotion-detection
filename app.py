import streamlit as st
import av
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load trained model
model = load_model("emotion_model.h5")

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# MTCNN face detector
detector = MTCNN()

st.set_page_config(page_title="Real-Time Emotion Detection", layout="centered")
st.title("🎭 Real-Time Emotion Detection (Webcam)")

class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = detector

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = img[:, :, ::-1]  # Convert BGR → RGB

        faces = self.detector.detect_faces(img_rgb)
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            face_img = img_rgb[y:y+h, x:x+w]
            if face_img.size != 0:
                face_pil = Image.fromarray(face_img).resize((48, 48)).convert("L")
                face_array = np.expand_dims(np.expand_dims(np.array(face_pil) / 255.0, axis=-1), axis=0)
                prediction = model.predict(face_array)
                emotion = EMOTIONS[np.argmax(prediction)]

                # Draw text & rectangle
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (36, 255, 12), 2)

        return img

webrtc_streamer(key="emotion", video_transformer_factory=EmotionTransformer)
