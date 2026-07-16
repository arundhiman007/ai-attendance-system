import cv2
import time
import threading
import numpy as np
import csv
from datetime import datetime
from collections import deque
import os

from app.camera.camera_stream import Camera
from app.detection.face_detector import FaceDetector
from app.recognition.recognizer import Recognizer


# 🔥 INIT SYSTEM
cam = Camera()
detector = FaceDetector()
rec = Recognizer()

# 🔥 CAMERA OPTIMIZATION
cam.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.cap.set(cv2.CAP_PROP_FPS, 30)

labels = {}
processing_flags = {}
buffers = {}
marked_attendance = set()

all_students = set(os.listdir("data/dataset"))

print("[INFO] Starting AI Attendance System...")


# ✅ MARK ATTENDANCE
def mark_attendance(name):
    if not name or name == "Unknown":
        return

    clean_name = name.split("_")[0]

    if clean_name in marked_attendance:
        return

    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([clean_name, "Present", datetime.now().strftime("%H:%M:%S")])

    marked_attendance.add(clean_name)
    print(f"[ATTENDANCE] {clean_name} Present")


# 🔥 FACE TRACKING
prev_boxes = []

def match_face_id(x, y, w, h):
    global prev_boxes

    for i, (px, py, pw, ph) in enumerate(prev_boxes):
        if abs(x - px) < 80 and abs(y - py) < 80:
            return i

    prev_boxes.append((x, y, w, h))
    return len(prev_boxes) - 1


# 🔥 ASYNC RECOGNITION
def recognize_async(face, face_id):
    try:
        name, dist = rec.recognize(face)

        if face_id not in buffers:
            buffers[face_id] = deque(maxlen=3)

        buffers[face_id].append((name, dist))

        names = [n for n, _ in buffers[face_id] if n]

        if not names:
            labels[face_id] = "Unknown"
            processing_flags[face_id] = False
            return

        most_common = max(set(names), key=names.count)

        if most_common and most_common != "Unknown":
            clean_name = most_common.split("_")[0]
            labels[face_id] = f"{clean_name} ✓"
            mark_attendance(most_common)
        else:
            labels[face_id] = "Unknown"

    except Exception as e:
        print("[ERROR]", e)
        labels[face_id] = "Unknown"

    processing_flags[face_id] = False


# 🔥 FPS CONTROL
prev_time = time.time()
frame_count = 0


# 🚀 MAIN LOOP
while True:
    frame = cam.get_frame()

    if frame is None:
        continue

    # 🔥 LIGHT IMPROVEMENT (CLAHE)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 🔥 SPEED BOOST (HALF FRAME)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    faces = detector.detect_faces(small_frame)
    prev_boxes = []

    if faces:
        for face_obj in faces[:4]:  # max 4 people

            box = face_obj["facial_area"]
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]

            # 🔥 SCALE BACK
            x *= 2
            y *= 2
            w *= 2
            h *= 2

            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face_id = match_face_id(x, y, w, h)

            if face_id not in processing_flags:
                processing_flags[face_id] = False
                labels[face_id] = "Scanning..."

            frame_count += 1

            # 🔥 CONTROL SPEED (important)
            if frame_count % 10 == 0 and not processing_flags[face_id]:
                processing_flags[face_id] = True

                threading.Thread(
                    target=recognize_async,
                    args=(face.copy(), face_id),
                    daemon=True
                ).start()

            # 🔥 COLOR BASED ON STATUS
            color = (0, 255, 0) if "✓" in labels[face_id] else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            cv2.putText(frame, labels[face_id], (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 2)

    # 🔥 FPS DISPLAY
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("AI Attendance System", frame)

    if cv2.waitKey(1) == 27:
        break


# 🔥 CLEANUP
cam.release()
cv2.destroyAllWindows()


# 🔥 MARK ABSENT STUDENTS
absent_students = []

for student in all_students:
    clean = student.split("_")[0]
    if clean not in marked_attendance:
        absent_students.append(clean)

with open("attendance.csv", "a", newline="") as f:
    writer = csv.writer(f)
    for name in absent_students:
        writer.writerow([name, "Absent", "-"])

print("[INFO] Attendance Complete")