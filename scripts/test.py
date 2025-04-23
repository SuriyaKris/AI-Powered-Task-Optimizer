import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000  # Sample rate
duration = 5  # seconds
filename = "test_recording.wav"

print("🎙️ Recording for 5 seconds... Speak now.")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
write(filename, fs, recording)
print(f"✅ Saved recording to {filename}")
