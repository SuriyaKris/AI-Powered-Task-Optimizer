import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.text_emotion_predictor import predict_emotion

def main():
    print("🎭 Multimodal Emotion Detection [Text-Based]")
    print("Type your message below (or type 'exit' to quit):\n")

    while True:
        text = input("You: ")
        if text.lower() == 'exit':
            print("👋 Exiting... Stay emotionally aware!")
            break

        emotion, confidence = predict_emotion(text)
        print(f"🤖 Detected Emotion: {emotion.capitalize()} (Confidence: {confidence:.2f})\n")

if __name__ == "__main__":
    main()
