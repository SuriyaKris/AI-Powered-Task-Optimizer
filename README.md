# AI-Powered-Task-Optimizer
# AI POWERED TASK OPTIMIZER

This is a real-time AI-powered system that detects employee emotions from **text**, **speech**, and **facial expressions**, and provides **personalized task recommendations** to enhance productivity and well-being.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
NOTE: not able to upload the datasets that was used so please train the Facial data with the FER dataset.
the emotions are mapped according to the FER-2013 dataset classes.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Text: BERT-based model from HuggingFace (fine-tuned for emotion detection)

Speech: Wav2Vec2 model for speech emotion recognition

Face: Custom ResNet18 model fine-tuned on FER-2013 dataset

Fusion: Weighted voting from all three modalities

## 🚀 Features

- ✅ Real-time **emotion detection** from:
  - 📝 Text input
  - 🎤 Voice (audio recording)
  - 📸 Facial expressions (webcam capture)
- ✅ **Weighted majority voting** to predict final emotion
- ✅ Emotion-based task recommendations using employee preferences
- ✅ Task selection logging with **SQLite** for future learning
- ✅ Alert system for repeated negative emotions (e.g., sad, angry)
- ✅ HR Dashboard to monitor employee mood trends and alerts
- ✅ Beautiful, responsive web UI built with Flask + HTML/CSS

---

## 🏗️ Project Structure

```bash
.
├── web/
│   ├── app.py                # Flask backend
│   ├── templates/            # HTML pages (home, dashboard)
│   ├── static/               # CSS styles
├── utils/
│   ├── text_emotion_predictor.py
│   ├── speech_emotion_predictor.py
│   ├── face_emotion_predictor.py
│   ├── fusion_predictor.py
│   ├── recommendation_engine.py
│   └── alert_checker.py
├── data/
│   ├── data_schema.py
│   └── emotion_task.db       # SQLite database
├── models/
│   ├── resnet18_face_emotion.pt
│   ├── bert-text-emotion/
│   └── wav2vec-speech-emotion/
└── README.md
