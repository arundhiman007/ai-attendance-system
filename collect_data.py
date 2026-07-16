import os
import cv2
import time
import sys
import numpy as np
import random

BASE_PATH = "data/dataset"
os.makedirs(BASE_PATH, exist_ok=True)

# CLI or input
if len(sys.argv) >= 3:
    name = sys.argv[1].strip()
    roll = sys.argv[2].strip()
else:
    name = input("Enter name: ").strip()
    roll = input("Enter roll: ").strip()

unique_id = f"{name}_{roll}"
dataset_path = os.path.join(BASE_PATH, unique_id)
os.makedirs(dataset_path, exist_ok=True)

print(f"[INFO] Collecting FACE dataset for: {name}")

# 🔥 Face detector (FAST)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

instructions = [
    "Look Straight",
    "Turn Left",
    "Turn Right",
    "Look Up",
    "Look Down",
    "Smile",
    "Normal Face"
]

cap = cv2.VideoCapture(0)

captured_images = []
instruction_index = 0
last_capture_time = 0


def augment_image(img):
    h, w = img.shape[:2]

    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # Draw box
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    if instruction_index < len(instructions):
        text = instructions[instruction_index]
    else:
        text = "Done"

    cv2.putText(frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Capture", frame)

    # 🔥 Capture ONLY FACE
    if len(faces) > 0 and instruction_index < len(instructions):

        if time.time() - last_capture_time > 1.5:

            x, y, w, h = faces[0]

            face = frame[y:y+h, x:x+w]

            face = cv2.resize(face, (112, 112))

            captured_images.append(face)

            print(f"[CAPTURED] {instructions[instruction_index]}")

            instruction_index += 1
            last_capture_time = time.time()

    if cv2.waitKey(1) == 27:
        break

    if instruction_index >= len(instructions):
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] Augmenting images...")

final_images = []

for img in captured_images:
    final_images.append(img)

    for _ in range(3):
        final_images.append(augment_image(img))

# Limit
final_images = final_images[:35]

# Save
for i, img in enumerate(final_images):
    cv2.imwrite(os.path.join(dataset_path, f"{i}.jpg"), img)

print(f"[SUCCESS] Saved {len(final_images)} images")