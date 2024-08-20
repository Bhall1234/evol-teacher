import json
import subprocess
import sys
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load tasks from JSON
with open('My_Work/new_architecture_v2/data/tasks.json') as f:
    tasks = json.load(f)['tasks']

@app.route("/")
def home():
    return render_template("index.html", tasks=tasks)

@app.route("/run_code", methods=["POST"])
def run_code():
    data = request.get_json()
    user_code = data.get("code", "")
    task_id = data.get("task_id", "")

    # Find the task
    task = next((t for t in tasks if t['task_id'] == int(task_id)), None)

    if not user_code or not task:
        return jsonify({"output": "Invalid code or task."}), 400

    try:
        result = subprocess.run([sys.executable, "-c", user_code], capture_output=True, text=True, check=True)
        user_output = result.stdout.strip()

        # Check against expected output
        if user_output == task["expected_output"]:
            return jsonify({"output": user_output, "result": "Correct"})
        else:
            return jsonify({"output": user_output, "result": "Incorrect", "expected_output": task["expected_output"]})

    except subprocess.CalledProcessError as e:
        return jsonify({"output": f"Error in code execution:\n{e.stderr.strip()}"}), 400

@app.route("/get_task", methods=["GET"])
def get_task():
    task_id = request.args.get("task_id", "")
    task = next((t for t in tasks if t['task_id'] == int(task_id)), None)
    if task:
        return jsonify(task)
    else:
        return jsonify({"error": "Task not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
