import pandas as pd
import os

# Load original dataset
df = pd.read_csv("dataset/train.csv")

# Create labels for eye images
labels = []

for i in range(len(df)):
    labels.append({
        "filename": f"eye_{i}.png",
        "emotion": df.loc[i, "emotion"]
    })

# Convert to DataFrame
labels_df = pd.DataFrame(labels)

# Save labels
output_path = "dataset/eye_labels.csv"
labels_df.to_csv(output_path, index=False)

print(f"Labels saved successfully!")
print(f"Total images: {len(labels_df)}")
print(f"Saved to: {output_path}")

print("\nEmotion distribution:")
print(labels_df["emotion"].value_counts().sort_index())