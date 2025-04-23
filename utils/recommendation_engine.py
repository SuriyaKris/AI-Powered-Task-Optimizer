import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import Counter
from data.data_schema import (
    get_preferred_tasks,
    get_employee_by_id,
    get_emotion_task_history_from_db,
    log_emotion_task  
)


# Define role-based task pool
role_task_pool = {
    "Software Engineer": [
        "Code Review", "Debug Issues", "Design System Architecture", 
        "Write Documentation", "Write Tests", "Pair Programming"
    ],
    "Data Analyst": [
        "Create Dashboards", "Analyze New Dataset", "Data Cleaning", 
        "Write Reports", "Team Brainstorm", "SQL Query Optimization"
    ],
    "Product Manager": [
        "Client Meeting", "Sprint Planning", "Write PRDs", 
        "Daily Standup", "Product Roadmap", "Review Product Specs"
    ],
    "UX Designer": [
        "Design New Features", "User Testing", "Wireframing", 
        "Update Style Guide", "Low-effort Mockups", "Review Feedback"
    ],
    "DevOps Engineer": [
        "Infrastructure Setup", "Automate Deployments", "Server Maintenance", 
        "Log Monitoring", "Write Scripts", "Document Setup"
    ]
}

def recommend_tasks(employee_id, emotion):
    # Step 1: Fetch emotion-task history from DB
    history = get_emotion_task_history_from_db(employee_id)
    emotion_tasks = [entry[1] for entry in history if entry[0].lower() == emotion.lower()]
    task_freq = Counter(emotion_tasks)

    # Step 2: Sort learned tasks with frequency
    learned_tasks_sorted = [f"{task} (x{count})" for task, count in task_freq.most_common()]
    learned_task_names = [task for task in task_freq]  # for filtering

    # Step 3: Add schema-preferred tasks not already learned
    preferred = get_preferred_tasks(employee_id, emotion)
    new_preferred = [task for task in preferred if task not in learned_task_names]

    # Step 4: Combine final recommended
    recommended = learned_tasks_sorted + new_preferred

    # Step 5: Get role-based task pool
    employee = get_employee_by_id(employee_id)
    role = employee["role"] if employee else None
    role_tasks = role_task_pool.get(role, [])

    # Step 6: Other tasks (excluding all recommended)
    others = [task for task in role_tasks if task not in learned_task_names + preferred]

    return recommended, others



def handle_task_selection(employee_id, emotion, recommended, others):
    print(f"\n📋 Recommended Tasks for Emotion [{emotion.title()}]:")
    for i, task in enumerate(recommended, 1):
        print(f"{i}. {task}")

    if others:
        print("\n🧰 Other Available Tasks:")
        for j, task in enumerate(others, len(recommended) + 1):
            print(f"{j}. {task}")

    try:
        choice = int(input("\nEnter the task number you want to select (0 to skip): "))
        if choice == 0:
            print("❌ No task was selected.")
            return
        task_list = recommended + others
        selected_task = task_list[choice - 1]
        log_emotion_task(employee_id, emotion, selected_task)
        print(f"✅ Task '{selected_task}' has been assigned and logged.")
    except (IndexError, ValueError):
        print("⚠️ Invalid choice. No task was selected.")
