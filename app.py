import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Eye Emotion Detector",
    page_icon="👁️",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------

st.markdown("""
<style>

.stApp {
    background-color: #fff7fb;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------

st.markdown(
    '<div class="title">👁️ Eye Emotion Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-Time Emotion Detection using CNN'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Emotion Classes
# -----------------------------

emotion_names = [
    "Angry 😠",
    "Disgust 🤢",
    "Fear 😨",
    "Happy 😊",
    "Sad 😢",
    "Surprise 😲",
    "Neutral 😐"
]

# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "models/eye_emotion_model.keras"
    )

model = load_model()

# -----------------------------
# Emotion Prediction Function
# -----------------------------

def predict_emotion(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Resize exactly like training
    face = cv2.resize(
        gray,
        (48, 48)
    )

    # Crop eye region
    eye_region = face[8:28, 4:44]

    # Resize
    eye_region = cv2.resize(
        eye_region,
        (48, 24),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    eye_region = (
        eye_region.astype("float32") / 255.0
    )

    # Add channel
    eye_region = np.expand_dims(
        eye_region,
        axis=-1
    )

    # Add batch
    eye_region = np.expand_dims(
        eye_region,
        axis=0
    )

    prediction = model.predict(
        eye_region,
        verbose=0
    )[0]

    index = int(
        np.argmax(prediction)
    )

    confidence = (
        float(prediction[index]) * 100
    )

    return emotion_names[index], confidence


# -----------------------------
# Webcam Transformer
# -----------------------------

class EmotionTransformer(VideoTransformerBase):

    def transform(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        emotion, confidence = predict_emotion(
            img
        )

        # Display emotion
        cv2.putText(
            img,
            f"Emotion: {emotion}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # Display confidence
        cv2.putText(
            img,
            f"Confidence: {confidence:.1f}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        return img


# -----------------------------
# Webcam Section
# -----------------------------

st.subheader("📷 Live Camera")

st.write(
    "Click START to open your webcam and "
    "detect emotions in real time."
)

webrtc_streamer(
    key="emotion-detector",
    video_transformer_factory=EmotionTransformer,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

# -----------------------------
# Information
# -----------------------------

st.markdown("---")

st.subheader("🧠 Supported Emotions")

cols = st.columns(7)

for col, emotion in zip(
    cols,
    emotion_names
):
    col.write(emotion)

st.info(
    "The model predicts emotions from the "
    "eye-region of the camera image."
)