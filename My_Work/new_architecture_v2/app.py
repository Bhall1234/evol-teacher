import subprocess
import sys
import os
import logging
from flask import Flask, request, render_template, jsonify
import json

app = Flask(__name__)

# Set up logging in OneDrive
log_path = os.path.join('C:\\Users\\benha\\OneDrive - The University of Nottingham\\Project\\Chat_Logs\\Participant', 'user_interactions_no_chatbot.log')
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load tasks from JSON
with open('My_Work/new_architecture_v2/data/tasks.json') as f:
    tasks = json.load(f)['tasks']

@app.route("/")
def home():
    logging.info("Home page accessed")
    return render_template("index.html", tasks=tasks)

@app.route("/run_code", methods=["POST"])
def run_code():
    data = request.get_json()
    user_code = data.get("code", "")
    task_id = data.get("task_id", "")

    logging.info(f"Run code requested for task_id: {task_id}")

    # Find the task
    task = next((t for t in tasks if t['task_id'] == int(task_id)), None)

    if not user_code or not task:
        logging.warning("Invalid code or task")
        return jsonify({"output": "Invalid code or task."}), 400

    try:
        result = subprocess.run([sys.executable, "-c", user_code], capture_output=True, text=True, check=True)
        user_output = result.stdout.strip()

        # Check against expected output
        if user_output == task["expected_output"]:
            logging.info("Code execution correct")
            return jsonify({"output": user_output, "result": "Correct"})
        else:
            logging.info("Code execution incorrect")
            return jsonify({"output": user_output, "result": "Incorrect", "expected_output": task["expected_output"]})

    except subprocess.CalledProcessError as e:
        logging.error(f"Error in code execution: {e.stderr.strip()}")
        return jsonify({"output": f"Error in code execution:\n{e.stderr.strip()}"}), 400

@app.route("/check_code", methods=["POST"])
def check_code():
    data = request.get_json()
    user_code = data.get("code", "")
    task_id = data.get("task_id", "")

    logging.info(f"Check code requested for task_id: {task_id}")

    # Find the task
    task = next((t for t in tasks if t['task_id'] == int(task_id)), None)

    if not user_code or not task:
        logging.warning("Invalid code or task")
        return jsonify({"output": "Invalid code or task."}), 400

    try:
        result = subprocess.run([sys.executable, "-c", user_code], capture_output=True, text=True, check=True)
        user_output = result.stdout.strip()

        # Check against expected output
        if user_output == task["expected_output"]:
            logging.info("Code execution correct")

            # Log user's code and mark it as correct
            logging.info(f"User's correct code for task {task_id}:\n{user_code}")

            reflection_question = task.get("reflection_question", "")

            return jsonify({
                "output": user_output,
                "result": "Correct",
                "reflection_question": reflection_question
            })
        else:
            logging.info("Code execution incorrect")
            return jsonify({
                "output": user_output,
                "result": "Incorrect",
                "expected_output": task["expected_output"],
                "reflection_question": ""
            })

    except subprocess.CalledProcessError as e:
        logging.error(f"Error in code execution: {e.stderr.strip()}")
        return jsonify({"output": f"Error in code execution:\n{e.stderr.strip()}"}), 400

@app.route("/submit_reflection", methods=["POST"])
def submit_reflection():
    data = request.get_json()
    reflection_response = data.get("reflection_response", "")
    task_id = data.get("task_id", "")

    logging.info(f"Reflection submitted for task_id: {task_id}")
    logging.info(f"User's reflection: {reflection_response}")

    return jsonify({"message": "Reflection submitted successfully."})

@app.route("/get_task", methods=["GET"])
def get_task():
    task_id = request.args.get("task_id", "")
    logging.info(f"Get task requested for task_id: {task_id}")
    task = next((t for t in tasks if t['task_id'] == int(task_id)), None)
    if task:
        return jsonify(task)
    else:
        logging.warning("Task not found")
        return jsonify({"error": "Task not found"}), 404
    
@app.route("/cant_figure_it_out", methods=["POST"])
def cant_figure_it_out():
    data = request.get_json()
    task_id = data.get("task_id", "")

    logging.info(f"User couldn't figure out task {task_id} and decided to move on.")

    return jsonify({"message": "It's okay! You've moved on to the next task."})

if __name__ == "__main__":
    logging.info("Starting Flask application")
    app.run(debug=True)
