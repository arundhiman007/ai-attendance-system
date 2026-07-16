import cv2
import time
from collections import deque
from app.camera.camera_stream import Camera
from app.detection.face_detector import FaceDetector
from app.recognition.recognizer import Recognizer

cam = Camera()
detector = FaceDetector()
rec = Recognizer()

buffer = deque(maxlen=5)

print("[INFO] Starting Recognition System...")

prev_time = time.time()

while True:
    frame = cam.get_frame()

    if frame is None:
        continue

    faces = detector.detect_faces(frame)

    label = "Scanning..."

    if faces:
        print("[DEBUG] Face detected")

        box = faces[0]["facial_area"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]

        face = frame[y:y+h, x:x+w]

        if face.size > 0:
            name, dist = rec.recognize(face)

            print(f"[DEBUG] Prediction: {name}, Distance: {dist:.4f}")

            if name:
                buffer.append(name)

                if buffer.count(name) >= 3:
                    if dist < 0.75:
                        label = name
                    elif dist < 0.85:
                        label = "Maybe " + name
                    else:
                        label = "Unknown"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    else:
        print("[DEBUG] No face detected")

    # 🔥 FPS CALCULATION
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # 🔥 DISPLAY INFO
    cv2.putText(frame, f"Status: {label}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("AI Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()