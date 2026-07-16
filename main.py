from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from app.recognition.recognizer import Recognizer

app = FastAPI()
rec = Recognizer()

@app.post("/attendance")
async def mark_attendance(file: UploadFile = File(...)):

    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    name = rec.recognize(img)

    return {"name": name}