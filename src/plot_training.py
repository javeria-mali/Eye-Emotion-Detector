import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# Load labels
# -----------------------------

df = pd.read_csv("dataset/eye_labels.csv")

X = []
y = []

print("Loading images...")

for i, row in df.iterrows():

    image_path = os.path.join(
        "dataset/eye_crops",
        row["filename"]
    )

    image = tf.keras.utils.load_img(
        image_path,
        color_mode="grayscale",
        target_size=(24, 48)
    )

    image = tf.keras.utils.img_to_array(image)
    image = image / 255.0

    X.append(image)
    y.append(int(row["emotion"]))

    if (i + 1) % 5000 == 0:
        print(f"Loaded {i + 1} / {len(df)}")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# -----------------------------
# Same validation split
# -----------------------------

_, X_val, _, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Load trained model
# -----------------------------

model = tf.keras.models.load_model(
    "models/eye_emotion_model.keras"
)

print("\nModel loaded successfully!")

# -----------------------------
# Predictions
# -----------------------------

predictions = model.predict(X_val, batch_size=64)

y_pred = np.argmax(predictions, axis=1)

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_val, y_pred)

emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=emotion_names
)

display.plot(
    xticks_rotation=45
)

plt.title("Eye Emotion Detector - Confusion Matrix")
plt.tight_layout()

# Save figure
os.makedirs("outputs", exist_ok=True)

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nConfusion matrix saved!")
print("Location: outputs/confusion_matrix.png")

plt.show()