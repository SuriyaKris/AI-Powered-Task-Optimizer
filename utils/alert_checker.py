# utils/alert_checker.py

import sqlite3
import os
from collections import Counter

# Pointing to the database
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Project root
db_path = os.path.join(BASE_DIR, 'data', 'emotion_task.db')

# Alert conditions
NEGATIVE_EMOTIONS = ["sad", "angry"]
ALERT_THRESHOLD = 3  # Number of repeated negative emotions to trigger alert

def check_employee_alert(employee_id):
    """Check if an employee needs HR attention based on past emotions."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT emotion FROM emotion_task_logs
        WHERE employee_id = ?
        ORDER BY timestamp DESC
        LIMIT 10
    ''', (employee_id,))
    recent_emotions = [row[0].lower() for row in cursor.fetchall()]
    conn.close()

    emotion_counts = Counter(recent_emotions)

    total_negative = sum(emotion_counts[emotion] for emotion in NEGATIVE_EMOTIONS)

    if total_negative >= ALERT_THRESHOLD:
        return True, total_negative
    else:
        return False, total_negative
