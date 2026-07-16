# 🎓 AI Attendance System using Computer Vision

> An AI-powered attendance management system that automatically recognizes students using Computer Vision and Face Recognition, eliminating the need for manual attendance while improving speed, accuracy, and efficiency.

---

# 📌 Overview

Traditional attendance systems are often time-consuming, manual, and prone to human errors. This project provides an intelligent attendance solution that uses face recognition to automatically identify students and mark their attendance in real time.

The system captures a student's facial dataset, generates face embeddings using **DeepFace (ArcFace)**, and performs real-time recognition through a webcam. Once a student is recognized, the system automatically records their attendance along with the timestamp.

This project was built to strengthen my understanding of **Computer Vision, Face Recognition, FastAPI, and AI application development** while designing a practical AI solution for educational environments.

---

# ✨ Features

✅ Face Dataset Collection

- Capture student images with multiple facial poses
- Guided image collection process
- Automatic dataset organization

---

✅ Image Augmentation

- Random rotation
- Brightness adjustment
- Contrast enhancement

This improves model robustness without requiring a large dataset.

---

✅ Face Encoding

- DeepFace (ArcFace)
- Automatic embedding generation
- Encoding storage for fast recognition

---

✅ Real-Time Face Recognition

- Live webcam detection
- Face localization
- Student identification
- Confidence-based prediction

---

✅ Automatic Attendance

- Marks attendance only once
- Stores attendance with date and timestamp
- Prevents duplicate entries

---

✅ User-Friendly Interface

- FastAPI backend
- Simple HTML/CSS/JavaScript frontend
- Easy interaction for attendance management

---

# 🏗️ System Workflow

```text
Student Registration
        │
        ▼
Capture Face Dataset
        │
        ▼
Image Augmentation
        │
        ▼
Generate Face Embeddings
        │
        ▼
Store Encodings
        │
        ▼
Start Camera
        │
        ▼
Detect Face
        │
        ▼
Recognize Student
        │
        ▼
Mark Attendance Automatically
```

---

# 🛠️ Tech Stack

## Programming Language

- Python

---

## Computer Vision

- OpenCV
- DeepFace
- ArcFace Model

---

## Backend

- FastAPI

---

## Frontend

- HTML
- CSS
- JavaScript

---

## Libraries

- NumPy
- Pickle
- CSV
- OS
- Datetime

---

# 📂 Project Structure

```text
AI-Attendance-System/

│
├── api/
│   └── api.py
│
├── app/
│   ├── camera/
│   ├── detection/
│   └── recognition/
│
├── frontend/
│   ├── index.html
│   ├── attendance.html
│   ├── success.html
│   ├── style.css
│   └── script.js
│
├── data/
│   ├── dataset/
│   └── encodings.pkl
│
├── assets/
│
├── collect_data.py
├── encode.py
├── run_system.py
├── requirements.txt
├── README.md
└── .gitignore

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/arundhiman007/ai-attendance-system-computer-vision.git
```

Move into the project directory.

```bash
cd ai-attendance-system-computer-vision
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# 🚀 How to Run

## Step 1 — Register Student

Collect the student's face dataset.

```bash
python collect_data.py
```

---

## Step 2 — Generate Face Encodings

Encode all registered students.

```bash
python encode.py
```

---

## Step 3 — Start Attendance System

Launch the recognition system.

```bash
python run_system.py
```

---

# 📊 Attendance Output

When a student is recognized:

- Student Name
- Attendance Status
- Date
- Time

are automatically stored in the attendance file.

Example:

| Name | Status | Time |
|------|--------|------|
| Ishant | Present | 09:12:45 |

---

# 💡 Challenges Solved

- Face recognition under different facial poses
- Dataset generation with limited images
- Improving recognition reliability using image augmentation
- Organizing a modular project architecture
- Real-time face detection and attendance recording
- Integrating backend and frontend components

---

# 📚 Key Learnings

This project strengthened my understanding of:

- Computer Vision
- Face Recognition
- Deep Learning Applications
- FastAPI Backend Development
- REST APIs
- Image Processing
- Python Application Development
- Project Architecture
- AI Solution Development

More importantly, it helped me understand how AI can be applied to solve real-world operational challenges through automation.

---

# 🚀 Future Enhancements

- Database Integration (PostgreSQL / MySQL)
- Face Anti-Spoofing
- Multi-Camera Support
- Cloud Deployment
- Student Dashboard
- Admin Dashboard
- Attendance Analytics
- Email Notifications
- Mobile Application

---

# 👨‍💻 Author

**Arun Dhiman**

AI Solutions Engineer

📧 Email: thearundhiman007@gmail.com

💼 LinkedIn: https://www.linkedin.com/in/arun-dhiman23/

🐙 GitHub: https://github.com/arundhiman007

---

## ⭐ If you found this project useful, consider giving it a star!
