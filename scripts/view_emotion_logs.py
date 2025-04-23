import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # goes to project root
db_path = os.path.join(BASE_DIR, 'data', 'emotion_task.db')

def show_logs():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM emotion_task_logs')
    rows = cursor.fetchall()
    conn.close()

    if rows:
        print("\n📘 Emotion-Task Logs:")
        for row in rows:
            print(f"ID: {row[0]} | Employee ID: {row[1]} | Emotion: {row[2]} | Task: {row[3]} | Timestamp: {row[4]}")
    else:
        print("🚫 No records found in the database.")

if __name__ == "__main__":
    show_logs()
    print("Reading DB at:", db_path)

