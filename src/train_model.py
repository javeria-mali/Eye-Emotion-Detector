import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# -----------------------------
# Settings
# -----------------------------

IMAGE_FOLDER = "dataset/eye_crops"
LABEL_FILE = "dataset/eye_labels.csv"

IMG_HEIGHT = 24
IMG_WIDTH = 48
NUM_CLASSES = 7

# -----------------------------
# Load labels
# -----------------------------

df = pd.read_csv(LABEL_FILE)

print("Total images:", len(df))
print("\nEmotion distribution:")
print(df["emotion"].value_counts().sort_index())

# -----------------------------
# Load images
# -----------------------------

X = []
y = []

print("\nLoading images...")

for i, row in df.iterrows():

    image_path = os.path.join(
        IMAGE_FOLDER,
        row["filename"]
    )

    image = tf.keras.utils.load_img(
        image_path,
        color_mode="grayscale",
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )

    image = tf.keras.utils.img_to_array(image)

    # Normalize pixels
    image = image / 255.0

    X.append(image)
    y.append(int(row["emotion"]))

    if (i + 1) % 5000 == 0:
        print(f"Loaded {i + 1} / {len(df)} images")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("\nData shape:", X.shape)
print("Labels shape:", y.shape)

# -----------------------------
# Train / validation split
# -----------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining images:", len(X_train))
print("Validation images:", len(X_val))

# -----------------------------
# CNN Model
# -----------------------------

model = models.Sequential([

    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

# -----------------------------
# Compile
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------

print("\nStarting CNN training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64
)

# -----------------------------
# Evaluate
# -----------------------------

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print("\n==============================")
print("MODEL TRAINING COMPLETED")
print("==============================")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Loss: {loss:.4f}")

# -----------------------------
# Save model
# -----------------------------

os.makedirs("models", exist_ok=True)

model.save("models/eye_emotion_model.keras")

print("\nModel saved successfully!")
print("Location: models/eye_emotion_model.keras")