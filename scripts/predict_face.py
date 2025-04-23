# scripts/predict_face.py

import cv2
import sys
import os

# Ensure utils folder is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.face_emotion_predictor import load_model, predict_emotion

def capture_face():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return None

    print("Press 'c' to capture image or 'q' to quit.")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                cap.release()
                cv2.destroyAllWindows()
                return face_img
            else:
                print("No face detected. Try again.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

def main():
    model = load_model()
    face_img = capture_face()
    if face_img is not None:
        emotion, confidence = predict_emotion(model, face_img)
        print(f"Detected Emotion: {emotion} (Confidence: {confidence:.2f})")
    else:
        print("Face capture failed.")

if __name__ == "__main__":
    main()
