import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/train.csv")

# First image
pixels = df.loc[0, "pixels"]

# Convert pixels into numbers
pixels = np.array(pixels.split(), dtype=np.uint8)

# Convert into 48x48 image
image = pixels.reshape(48, 48)

# Get emotion label
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

emotion = emotion_labels[df.loc[0, "emotion"]]

# Display image
plt.imshow(image, cmap="gray")
plt.title(f"Emotion: {emotion}")
plt.axis("off")
plt.show()