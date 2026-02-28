"""
Real-Time Face Detection App
Uses OpenCV and Haar Cascades for face detection via webcam
"""

import cv2
import sys
from datetime import datetime


def load_cascades():
    """Load face and eye cascade classifiers"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if face_cascade.empty():
        print("Error: Could not load face cascade classifier.")
        sys.exit(1)

    return face_cascade, eye_cascade


def detect_faces(frame, face_cascade, eye_cascade, detect_eyes=True):
    """Detect faces (and optionally eyes) in a frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve detection in low light

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    face_count = 0

    for (x, y, w, h) in faces:
        face_count += 1

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Face label
        label = f"Face {face_count}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detect eyes within face region
        if detect_eyes:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(20, 20)
            )
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    return frame, face_count


def draw_ui(frame, face_count, fps, detect_eyes, recording):
    """Draw UI overlays on the frame"""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Title
    cv2.putText(frame, "Real-Time Face Detection App", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Bottom info bar
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Face count
    face_text = f"Faces: {face_count}"
    color = (0, 255, 0) if face_count > 0 else (0, 0, 255)
    cv2.putText(frame, face_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (200, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Eye detection status
    eye_status = "Eyes: ON" if detect_eyes else "Eyes: OFF"
    cv2.putText(frame, eye_status, (350, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Recording indicator
    if recording:
        cv2.circle(frame, (w - 30, 22), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 70, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Controls hint
    controls = "[Q] Quit  [S] Screenshot  [E] Toggle Eyes  [R] Record"
    cv2.putText(frame, controls, (10, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    return frame


def main():
    print("=" * 50)
    print("  Real-Time Face Detection App")
    print("=" * 50)
    print("\nStarting camera...")

    face_cascade, eye_cascade = load_cascades()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check if camera is connected.")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera opened successfully!")
    print("\nControls:")
    print("  Q       - Quit the app")
    print("  S       - Save screenshot")
    print("  E       - Toggle eye detection")
    print("  R       - Start/Stop recording")
    print("\nPress any key in the window to start...\n")

    # State variables
    detect_eyes = True
    recording = False
    video_writer = None
    screenshot_count = 0

    # FPS calculation
    fps = 0
    frame_count = 0
    start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        # Flip frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        # Detect faces
        frame, face_count = detect_faces(frame, face_cascade, eye_cascade, detect_eyes)

        # Calculate FPS
        frame_count += 1
        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = cv2.getTickCount()

        # Draw UI
        frame = draw_ui(frame, face_count, fps, detect_eyes, recording)

        # Write to video if recording
        if recording and video_writer:
            video_writer.write(frame)

        # Show frame
        cv2.imshow("Real-Time Face Detection App", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC to quit
            print("\nQuitting...")
            break

        elif key == ord('s'):  # Screenshot
            screenshot_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

        elif key == ord('e'):  # Toggle eye detection
            detect_eyes = not detect_eyes
            status = "ON" if detect_eyes else "OFF"
            print(f"Eye detection: {status}")

        elif key == ord('r'):  # Toggle recording
            if not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.avi"
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
                recording = True
                print(f"Recording started: {filename}")
            else:
                recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                print("Recording stopped.")

    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("App closed successfully.")


if __name__ == "__main__":
    main()
