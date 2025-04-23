# web/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify, redirect, url_for
import base64
import io
import tempfile
import numpy as np
from PIL import Image
from pydub import AudioSegment

from utils.text_emotion_predictor import predict_emotion as predict_text_emotion
from utils.speech_emotion_predictor import predict_emotion_from_audio
from utils.face_emotion_predictor import load_model, predict_emotion as predict_face_emotion
from utils.fusion_predictor import predict_emotion as fuse_emotions
from utils.recommendation_engine import recommend_tasks
from utils.alert_checker import check_employee_alert  # 🆕 Import alert checker
from data.data_schema import log_emotion_task

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Load the face emotion model once
face_model = load_model()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            employee_id = request.form.get('employee_id', '').strip()
            if not employee_id:
                return render_template('home.html', error="Please enter Employee ID.")

            text_input = request.form.get('text_input')
            audio_base64 = request.form.get('audio_data')
            image_base64 = request.form.get('image_data')

            text_emotion = speech_emotion = facial_emotion = None

            # Text emotion
            if text_input:
                text_emotion, _ = predict_text_emotion(text_input)

            # Speech emotion
            if audio_base64:
                audio_data = base64.b64decode(audio_base64.split(',')[1])
                with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmpfile:
                    tmpfile.write(audio_data)
                    temp_webm_path = tmpfile.name

                temp_wav_path = temp_webm_path.replace('.webm', '.wav')
                AudioSegment.from_file(temp_webm_path).export(temp_wav_path, format="wav")
                os.remove(temp_webm_path)

                speech_emotion, _ = predict_emotion_from_audio(temp_wav_path)
                os.remove(temp_wav_path)

            # Facial emotion
            if image_base64:
                image_data = base64.b64decode(image_base64.split(',')[1])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                image = image.resize((224, 224))  # resize to FER-2013 size
                image_np = np.array(image)
                facial_emotion, _ = predict_face_emotion(face_model, image_np)

            final_emotion = fuse_emotions(text_emotion, speech_emotion, facial_emotion)

            recommended, others = recommend_tasks(employee_id, final_emotion)

            return render_template('home.html',
                                   employee_id=employee_id,
                                   emotion=final_emotion,
                                   recommended=recommended,
                                   others=others)

        except Exception as e:
            return render_template('home.html', error=str(e))

    return render_template('home.html')

@app.route('/log_task', methods=['POST'])
def log_task():
    try:
        data = request.get_json()
        employee_id = data['employee_id']
        emotion = data['emotion']
        task = data['task']
        log_emotion_task(employee_id, emotion, task)

        # 🆕 After logging, check for alert
        alert, count = check_employee_alert(employee_id)

        if alert:
            return jsonify({'status': 'alert', 'message': f'⚠️ Alert: Employee {employee_id} has shown {count} negative emotions recently!'}), 200
        else:
            return jsonify({'status': 'success', 'message': 'Task logged successfully.'}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

import sqlite3

@app.route('/dashboard')
def dashboard():
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        db_path = os.path.join(BASE_DIR, 'data', 'emotion_task.db')

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT employee_id, emotion, task, timestamp
            FROM emotion_task_logs
            ORDER BY timestamp DESC
        ''')
        logs = cursor.fetchall()
        conn.close()

        # Group logs per employee
        employees = {}
        for emp_id, emotion, task, timestamp in logs:
            if emp_id not in employees:
                employees[emp_id] = []
            employees[emp_id].append({
                'emotion': emotion,
                'task': task,
                'timestamp': timestamp
            })

        # Check which employees have alerts
        alerts = {}
        for emp_id in employees:
            from utils.alert_checker import check_employee_alert
            is_alert, _ = check_employee_alert(emp_id)
            alerts[emp_id] = is_alert

        return render_template('dashboard.html', employees=employees, alerts=alerts)

    except Exception as e:
        return f"Error loading dashboard: {e}"


if __name__ == '__main__':
    app.run(debug=True)
