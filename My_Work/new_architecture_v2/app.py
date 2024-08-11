# new version finding programming keywords in the explanation and fetching related code examples
import os
import random
import logging
import re
from flask import Flask, request, render_template, jsonify
from src.utils import load_dataset
from src.explanation_generation import generate_explanation
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import subprocess
import sys
import spacy

app = Flask(__name__)

# Set up logging
log_path = os.path.join(os.getcwd(), 'My_Work', 'new_architecture_v2', 'src', 'logs', 'chat_interactions.log')
os.makedirs(os.path.dirname(log_path), exist_ok=True)  # Ensure the log directory exists
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

# Load datasets
correct_code_examples = load_dataset('./My_Work/new_architecture_v2/data/code_examples.json')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form["question"]
    logging.info(f"User question: {user_question}")

    # Check if "python" is in the question, if not, append it.
    if "python" not in user_question.lower():
        user_question += " (programming Language: python)"
        logging.info(f"Modified user question: {user_question}")
    
    # Generate explanation using LLM
    explanation = generate_explanation(user_question, "TheBloke/CodeLlama-13B-Instruct-GGUF")
    
    # Extract the initial part of the explanation for keyword extraction
    initial_explanation = extract_initial_explanation(explanation)
    
    # Extract keywords from the initial explanation, and find related code examples from the dataset based on these keywords.
    explanation_keywords = extract_keywords_from_text(initial_explanation)
    
    # Match the explanation keywords with incorrect code examples
    incorrect_code_data = get_related_code_by_keywords(explanation_keywords, correct_code_examples)
    
    # Log the task_id to ensure it's being set correctly
    logging.info(f"Task ID: {incorrect_code_data.get('task_id', 'No task_id found')}")
    
    # Apply syntax highlighting to the explanation and incorrect code
    formatted_explanation = format_code_snippets(explanation)  # Use the full explanation for display
    formatted_incorrect_code = highlight(incorrect_code_data["incorrect_code"], PythonLexer(), HtmlFormatter(noclasses=True))
    
    logging.info(f"Generated explanation: {explanation}")
    logging.info(f"Incorrect code: {incorrect_code_data['incorrect_code']}")
    
    return render_template("index.html", question=user_question, explanation=formatted_explanation, 
                           incorrect_code=formatted_incorrect_code, task_description=incorrect_code_data["task_description"],
                           hint=incorrect_code_data["description"], detailed_explanation=incorrect_code_data["explanation"],
                           task_id=incorrect_code_data["task_id"])

@app.route("/run_code", methods=["POST"])
def run_code():
    data = request.get_json()
    code = data["code"]
    try:
        # Use the Python interpreter from the virtual environment
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = e.stderr
    return jsonify({"output": output})

@app.route("/check_code", methods=["POST"])
def check_code():
    data = request.get_json()
    user_code = data["code"]
    task_id = data["task_id"]
    
    # Log the task_id to ensure it's being received correctly
    logging.info(f"Received task ID: {task_id} (type: {type(task_id)})")
    
    # Find the correct code example based on the task ID
    correct_example = next((ex for ex in correct_code_examples["examples"] if str(ex["task_id"]) == str(task_id)), None)
    
    if not correct_example:
        logging.error("No matching task found.")
        return jsonify({"result": "No matching task found."}), 404
    
    correct_code = correct_example["correct_code"]
    
    # Compare the user's code with the correct code
    if user_code.strip() == correct_code.strip():
        result = "Correct"
    else:
        result = "Incorrect"
    
    return jsonify({"result": result})

def extract_initial_explanation(explanation):
    # Split the explanation into sentences using both '.' and ':', these are the most common sentence delimiters in the explanations.
    sentences = re.split(r'[.:]\s*', explanation)
    initial_part = '. '.join(sentences[:1])  # Adjust the number of sentences as needed
    return initial_part

def extract_keywords_from_text(text):
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    
    # Define a list of programming keywords
    programming_keywords = ["for", "while","for loop","while loop","`while`","`for`","`if`","if","`def`","else", "elif", "def", "class", "import", "try", "except", "with", "return"]
    
    # Check for the presence of these keywords in the user's question
    detected_programming_keywords = [kw for kw in text.lower().split() if kw in programming_keywords]
    
    # Combine the extracted keywords with the detected programming keywords
    combined_keywords = set(keywords + detected_programming_keywords)
    
    logging.info(f"Extracted keywords from text: {combined_keywords}")
    return combined_keywords

def get_related_code_by_keywords(keywords, correct_code_examples):
    logging.info(f"Matching keywords: {keywords}")
    
    # Filter examples by label
    filtered_examples = [ex for ex in correct_code_examples["examples"] if any(label in ex["label"] for label in keywords)]

    if not filtered_examples:
        return {"incorrect_code": "No related code examples found.", "task_description": "", "description": "", "explanation": "", "task_id": "N/A"}

    best_match = random.choice(filtered_examples)
    return best_match

def format_code_snippets(response):
    # This regex finds all code blocks wrapped in triple backticks
    code_block_pattern = re.compile(r'```(python)?\n(.*?)\n```', re.DOTALL)
    
    def replace_code_block(match):
        # Capture the code block and remove the 'python' if present
        code = match.group(2).strip()
        highlighted_code = highlight(code, PythonLexer(), HtmlFormatter(noclasses=True))
        return f"<pre><code>{highlighted_code}</code></pre>"

    formatted_response = code_block_pattern.sub(replace_code_block, response)
    return formatted_response

if __name__ == "__main__":
    app.run(debug=True)