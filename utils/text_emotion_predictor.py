from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from collections import defaultdict

# Load model/tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/text_model")
model = AutoModelForSequenceClassification.from_pretrained("models/text_model")

# Load correct label order from model
HF_LABELS = list(model.config.id2label.values())

# FER-2013 Mapping
FER_MAPPING = {
    'angry': ['anger', 'annoyance', 'disapproval'],
    'disgust': ['disgust'],
    'fear': ['fear', 'nervousness'],
    'happy': ['joy', 'love', 'amusement', 'gratitude', 'optimism', 'pride', 'relief', 'admiration', 'approval', 'caring', 'excitement'],
    'sad': ['sadness', 'grief', 'disappointment', 'remorse', 'embarrassment'],
    'surprise': ['surprise', 'realization'],
    'neutral': ['neutral', 'confusion', 'curiosity', 'desire']
}

# Reverse mapping: HF → FER
HF_TO_FER = {}
for fer_label, hf_list in FER_MAPPING.items():
    for hf_label in hf_list:
        HF_TO_FER[hf_label] = fer_label

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()

    fer_probs = defaultdict(float)
    for i, prob in enumerate(probs):
        hf_label = HF_LABELS[i]
        fer_label = HF_TO_FER.get(hf_label)
        if fer_label:
            fer_probs[fer_label] += prob.item()

    # Debug output
    print("\n🔍 DEBUG INFO")
    for label in sorted(fer_probs, key=fer_probs.get, reverse=True):
        print(f"{label.capitalize():<10} → {fer_probs[label]:.4f}")
    print()

    top_fer = max(fer_probs, key=fer_probs.get)
    return top_fer, fer_probs[top_fer]
