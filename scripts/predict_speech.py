import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
from utils.speech_emotion_predictor import predict_emotion_from_audio
import sys
import os

# Add project root to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)

from utils.speech_emotion_predictor import predict_emotion_from_audio


def record_audio(filename, duration=5, fs=16000):
    print("🎙️ Recording... Speak now.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    print("✅ Recording complete.")

def main():
    # Create a temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        temp_audio_path = tmpfile.name

    try:
        record_audio(temp_audio_path, duration=5)
        emotion, confidence = predict_emotion_from_audio(temp_audio_path)
        print(f"\n🔍 Detected Emotion: {emotion.capitalize()} ({confidence * 100:.2f}% confidence)")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == "__main__":
    main()
