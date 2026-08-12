import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
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
    '<div class="subtitle">Emotion Detection using CNN</div>',
    unsafe_allow_html=True
)

st.markdown("---")

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
# Prediction Function
# -----------------------------

def predict_emotion(image):

    # Convert image to NumPy
    image = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize
    face = cv2.resize(gray, (48, 48))

    # Crop eye region
    eye_region = face[8:28, 4:44]

    # Resize eye region
    eye_region = cv2.resize(
        eye_region,
        (48, 24),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    eye_region = eye_region.astype("float32") / 255.0

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

    # Prediction
    prediction = model.predict(
        eye_region,
        verbose=0
    )[0]

    index = int(np.argmax(prediction))

    confidence = float(prediction[index]) * 100

    return emotion_names[index], confidence


# =====================================================
# LIVE CAMERA
# =====================================================

st.subheader("📷 Live Camera")

st.write(
    "Click START to open your webcam and detect emotions "
    "in real time."
)


class EmotionProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Convert BGR to RGB
        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # Prediction
        emotion, confidence = predict_emotion(rgb)

        # Display emotion
        cv2.putText(
            img,
            f"Emotion: {emotion}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # Display confidence
        cv2.putText(
            img,
            f"Confidence: {confidence:.1f}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


webrtc_streamer(
    key="emotion-detector",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)


# =====================================================
# UPLOAD IMAGE
# =====================================================

st.markdown("---")

st.subheader("📤 Upload an Image")

st.write(
    "Upload a face image and the CNN model will predict "
    "the emotion from the eye region."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    emotion, confidence = predict_emotion(image)

    with col2:
        st.subheader("Prediction")

        st.success(
            f"Emotion: {emotion}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.1f}%"
        )


# =====================================================
# SUPPORTED EMOTIONS
# =====================================================

st.markdown("---")

st.subheader("🧠 Supported Emotions")

cols = st.columns(7)

for col, emotion in zip(cols, emotion_names):
    col.write(emotion)

st.info(
    "The CNN model predicts one of seven emotions "
    "from the processed eye-region image."
)