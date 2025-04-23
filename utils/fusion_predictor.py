from collections import defaultdict

FER_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MODALITY_WEIGHTS = {"text": 0.3, "speech": 0.3, "face": 0.4}

def predict_emotion(text_emotion, speech_emotion, face_emotion):
    # Define your weights
    weights = {
        'text': 1.0,
        'speech': 1.2,
        'face': 1.5
    }

    scores = {}

    for emotion in [text_emotion, speech_emotion, face_emotion]:
        if emotion not in scores:
            scores[emotion] = 0

    scores[text_emotion] += weights['text']
    scores[speech_emotion] += weights['speech']
    scores[face_emotion] += weights['face']

    final_emotion = max(scores, key=scores.get)
    return final_emotion

