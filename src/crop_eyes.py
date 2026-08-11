import cv2
import pandas as pd
import numpy as np
import os

# Load dataset
df = pd.read_csv("dataset/train.csv")

# Output folder
output_folder = "dataset/eye_crops"
os.makedirs(output_folder, exist_ok=True)

# Process the complete dataset
for i in range(len(df)):

    # Convert pixels into image
    pixels = np.array(
        df.loc[i, "pixels"].split(),
        dtype=np.uint8
    )

    face = pixels.reshape(48, 48)

    # Eye region
    eye_region = face[8:28, 4:44]

    # Resize to 48x24
    eye_region = cv2.resize(
        eye_region,
        (48, 24),
        interpolation=cv2.INTER_AREA
    )

    # Save image
    output_path = os.path.join(
        output_folder,
        f"eye_{i}.png"
    )

    cv2.imwrite(output_path, eye_region)

    # Show progress
    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1} / {len(df)} images")

print(f"\n{len(df)} eye-region images created successfully!")