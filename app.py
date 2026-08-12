import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

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
    'Emotion Detection using CNN'
    '</div>',
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

    # Convert PIL image to NumPy
    image = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize to 48x48
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

    return emotion_names[index], confidence, eye_region


# -----------------------------
# Upload Image
# -----------------------------

st.subheader("📷 Upload an Image")

st.write(
    "Upload a face image and the model will predict "
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

    emotion, confidence, eye_region = predict_emotion(image)

    with col2:
        st.subheader("Prediction")

        st.success(
            f"Emotion: {emotion}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.1f}%"
        )

        # Display processed eye region
        eye_display = eye_region[0, :, :, 0]

        st.image(
            eye_display,
            caption="Processed Eye Region",
            use_container_width=True
        )

# -----------------------------
# Supported Emotions
# -----------------------------

st.markdown("---")

st.subheader("🧠 Supported Emotions")

cols = st.columns(7)

for col, emotion in zip(cols, emotion_names):
    col.write(emotion)

st.info(
    "The CNN model predicts one of seven emotions "
    "from the processed eye-region image."
)