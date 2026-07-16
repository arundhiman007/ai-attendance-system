import os
import pickle
import cv2
import numpy as np
from deepface import DeepFace

DATASET_PATH = "data/dataset"
ENCODINGS_PATH = "data/encodings.pkl"

known_encodings = []
known_names = []

print("[INFO] Encoding faces using DeepFace (ArcFace)...")

for person_name in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"[ENCODING] {person_name}")

    count = 0

    for image_name in os.listdir(person_path):

        path = os.path.join(person_path, image_name)

        img = cv2.imread(path)
        if img is None:
            continue

        try:
            result = DeepFace.represent(
                img_path=img,
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False
            )

            if result and isinstance(result, list):
                emb = np.array(result[0]["embedding"])
                emb = emb / np.linalg.norm(emb)

                known_encodings.append(emb)
                known_names.append(person_name)
                count += 1

        except Exception as e:
            print(f"[ERROR] {image_name}")

    print(f"[DONE] {person_name} → {count} images")

# SAVE
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump({
        "encodings": known_encodings,
        "names": known_names
    }, f)

print("[SUCCESS] Encoding complete!")