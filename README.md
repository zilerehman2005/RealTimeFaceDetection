# Real-Time Face Detection App

A Python app that detects faces and eyes in real-time using your webcam.

## Features
- Real-time face detection using Haar Cascades
- Optional eye detection inside each face
- FPS display
- Screenshot capture
- Video recording
- Mirror mode (natural selfie view)

---

## Setup & Run in VS Code

### Step 1: Install Python
Make sure Python 3.8+ is installed.
Download from: https://www.python.org/downloads/
✅ Check "Add Python to PATH" during installation.

---

### Step 2: Open Project in VS Code
1. Open VS Code
2. Go to **File → Open Folder**
3. Select the folder containing `face_detection.py`

---

### Step 3: Open Terminal in VS Code
Press **Ctrl + `** (backtick) to open the integrated terminal.

---

### Step 4: Create a Virtual Environment (Recommended)
```bash
python -m venv venv
```

Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You'll see `(venv)` appear in the terminal prompt.

---

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

---

### Step 6: Run the App
```bash
python face_detection.py
```

A window will open showing your webcam feed with face detection boxes.

---

## Controls (Inside the App Window)

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit the app |
| `S` | Save a screenshot (saves as .jpg) |
| `E` | Toggle eye detection ON/OFF |
| `R` | Start/Stop video recording (saves as .avi) |

---

## Troubleshooting

**Camera not opening?**
- Make sure no other app (Zoom, Teams) is using the camera
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the code

**Low FPS?**
- Turn off eye detection by pressing `E`
- Lower resolution in the code (change 1280x720 to 640x480)

**Module not found error?**
- Make sure your virtual environment is activated
- Re-run `pip install -r requirements.txt`

---

## File Structure
```
face_detection_app/
├── face_detection.py    ← Main application
├── requirements.txt     ← Dependencies
└── README.md            ← This file
```
