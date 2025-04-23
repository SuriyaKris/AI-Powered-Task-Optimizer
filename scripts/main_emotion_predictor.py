import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.text_emotion_predictor import predict_emotion as predict_text_emotion
from utils.speech_emotion_predictor import predict_emotion_from_audio
from utils.face_emotion_predictor import load_model, predict_emotion as predict_face_emotion
from utils.fusion_predictor import predict_emotion as fuse_emotions
from utils.recommendation_engine import recommend_tasks, handle_task_selection

import sounddevice as sd
from scipy.io.wavfile import write
import cv2
import tempfile

# --- 0. Ask for Employee ID ---
employee_id = input("Enter your employee ID: ")

# --- 1. Text Emotion ---
text_input = input("Enter your text: ")
text_emotion, _ = predict_text_emotion(text_input)

# --- 2. Speech Emotion ---
print("Recording audio for 5 seconds...")
fs = 16000
seconds = 5
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
    temp_audio_path = tmpfile.name
    write(temp_audio_path, fs, recording)

speech_emotion, _ = predict_emotion_from_audio(temp_audio_path)

# --- 3. Facial Emotion ---
model = load_model()
cap = cv2.VideoCapture(0)
print("Capturing face. Press 's' to snap.")

face_image = None
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("Capture Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        face_image = frame
        break

cap.release()
cv2.destroyAllWindows()

face_emotion, _ = predict_face_emotion(model, face_image)

# --- Fusion ---
print(f"\nText Emotion: {text_emotion}")
print(f"Speech Emotion: {speech_emotion}")
print(f"Face Emotion: {face_emotion}")

final_emotion = fuse_emotions(text_emotion, speech_emotion, face_emotion)
print(f"\n🧠 Final Emotion Prediction: {final_emotion}")

# --- Task Recommendation ---
recommended, others = recommend_tasks(employee_id, final_emotion)
handle_task_selection(employee_id, final_emotion, recommended, others)
