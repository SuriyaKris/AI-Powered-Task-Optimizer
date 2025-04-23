# utils/face_emotion_predictor.py

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2

# Emotion labels corresponding to FER-2013
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_model(model_path="models/fer_resnet18.pth"):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 7)  # 7 classes for FER-2013
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_emotion(model, face_image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet norms
                             std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(face_image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return emotion_labels[predicted.item()], confidence.item()
