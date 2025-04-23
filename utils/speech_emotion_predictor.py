import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import os

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

feature_extractor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
model = AutoModelForAudioClassification.from_pretrained("superb/wav2vec2-base-superb-er")


# Save locally
SAVE_DIR = "models/speech_model"
feature_extractor.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

LABELS = list(model.config.id2label.values())

def predict_emotion_from_audio(audio_path):
    speech_array, sampling_rate = torchaudio.load(audio_path)
    speech_array = speech_array.squeeze()

    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech_array = resampler(speech_array)

    inputs = feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1)[0][predicted_id].item()

    predicted_emotion = LABELS[predicted_id]
    return predicted_emotion, confidence
