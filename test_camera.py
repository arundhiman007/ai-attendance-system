import cv2
from app.camera.camera_stream import Camera
from app.detection.face_detector import FaceDetector

cam = Camera()
detector = FaceDetector()

frame_count = 0
last_bbox = None

print("[INFO] Optimized DeepFace Detection Running...")

while True:
    frame = cam.get_frame()

    if frame is None:
        continue

    frame_count += 1

    # 🔥 VERY IMPORTANT → run DeepFace only every 5 frames
    if frame_count % 5 == 0:
        faces = detector.detect_faces(frame)

        if faces:
            last_bbox = faces[0]["facial_area"]

    # 🔥 Draw cached result (NO heavy computation)
    if last_bbox:
        x = last_bbox["x"]
        y = last_bbox["y"]
        w = last_bbox["w"]
        h = last_bbox["h"]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Optimized DeepFace Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()