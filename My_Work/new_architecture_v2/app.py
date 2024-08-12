import logging
import random
import re
import os
import tempfile
import subprocess
import sys
from flask import Flask, request, render_template, jsonify
from src.utils import load_dataset
from src.explanation_generation import generate_explanation
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import spacy
from fuzzywuzzy import fuzz

app = Flask(__name__)

# Set up logging
log_path = os.path.join(os.getcwd(), 'My_Work', 'new_architecture_v2', 'src', 'logs', 'chat_interactions.log')
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

# Load datasets
correct_code_examples = load_dataset('./My_Work/new_architecture_v2/data/code_examples.json')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Synonyms for better keyword matching
synonyms = {
    "append": ["add", "insert", "push"],
    "loop": ["iterate", "repeat", "for loop", "while loop", "looping", "iteration", "repetition", "loop through", "loop over", "looping through", "looping over"],
    "condition": ["if", "else", "elif", "ternary", "ternary operator", "conditional", "ternary condition"],
    "variable": ["var", "name", "identifier"],
    "list": ["array"],
    "dictionary": ["dict", "map", "hash map", "hash table", "key-value"],
    "tuple": ["pair", "ordered pair"],
    "range": ["sequence"],
    "open": ["read"],
    "write": ["save"],
    "import": ["include"], 
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form["question"]
    logging.info(f"User question: {user_question}")

    # Append "python" if not present in the question
    if "python" not in user_question.lower():
        user_question += " (programming Language: python)"
        logging.info(f"Modified user question: {user_question}")
    
    # Generate explanation using LLM
    explanation = generate_explanation(user_question, "TheBloke/CodeLlama-13B-Instruct-GGUF")
    logging.info(f"Generated explanation: {explanation}")
    
    initial_explanation = extract_initial_explanation(explanation)
    logging.info(f"Initial part of the explanation: {initial_explanation}")
    
    # Extract keywords from the user question and initial explanation
    user_keywords = extract_programming_keywords(user_question)
    explanation_keywords = extract_programming_keywords(initial_explanation)
    combined_keywords = user_keywords.union(explanation_keywords)
    logging.info(f"Extracted user keywords: {user_keywords}")
    logging.info(f"Extracted explanation keywords: {explanation_keywords}")
    logging.info(f"Combined keywords: {combined_keywords}")
    
    # Match the combined keywords with incorrect code examples
    incorrect_code_data = get_related_code_by_keywords(combined_keywords, correct_code_examples)
    logging.info(f"Selected incorrect code example: {incorrect_code_data}")
    
    # Log the task_id to ensure it's being set correctly
    logging.info(f"Task ID: {incorrect_code_data.get('task_id', 'No task_id found')}")
    
    # Apply syntax highlighting to the explanation and incorrect code
    formatted_explanation = format_code_snippets(explanation)
    formatted_incorrect_code = highlight(incorrect_code_data.get("incorrect_code", "No incorrect code found."), PythonLexer(), HtmlFormatter(noclasses=True))
    formatted_correct_code = highlight(incorrect_code_data.get("correct_code", "No correct code found."), PythonLexer(), HtmlFormatter(noclasses=True))
    
    return render_template("index.html", question=user_question, explanation=formatted_explanation, 
                           incorrect_code=formatted_incorrect_code, correct_code=formatted_correct_code,
                           task_description=incorrect_code_data.get("task_description", "No task description found."),
                           hint=incorrect_code_data.get("description", "No hint found."), 
                           detailed_explanation=incorrect_code_data.get("explanation", "No detailed explanation found."),
                           task_id=incorrect_code_data.get("task_id", "No task ID found."))

@app.route("/run_code", methods=["POST"])
def run_code():
    data = request.get_json()
    code = data["code"]
    logging.info(f"Running user code:\n{code}")
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        output = result.stdout
        logging.info(f"Code output: {output}")
    except subprocess.CalledProcessError as e:
        output = e.stderr
        logging.error(f"Code execution error: {output}")
    return jsonify({"output": output})

@app.route("/check_code", methods=["POST"])
def check_code():
    try:
        data = request.get_json()
        user_code = data["code"]
        task_id = data["task_id"]
        logging.info(f"Received task ID: {task_id} (type: {type(task_id)})")
        
        # Find the correct code example based on the task ID
        correct_example = find_correct_example(task_id, correct_code_examples)
        if not correct_example:
            logging.error("No matching task found.")
            return jsonify({"result": "No matching task found."}), 404
        
        expected_output = correct_example["expected_output"]
        logging.info(f"Expected output: {expected_output}")
        logging.info(f"User code:\n{user_code}")
        
        try:
            result = subprocess.run([sys.executable, "-c", user_code], capture_output=True, text=True, check=True)
            user_output = result.stdout
            logging.info(f"User code output: {user_output}")
        except subprocess.CalledProcessError as e:
            user_output = e.stderr
            logging.error(f"User code execution error: {user_output}")
        
        result = "Correct" if user_output.strip() == expected_output.strip() else "Incorrect"
        logging.info(f"Code check result: {result}")
        static_analysis_result = run_static_analysis(user_code)
        
        return jsonify({
            "result": result,
            "static_analysis": static_analysis_result,
        })
    except Exception as e:
        logging.error(f"Error in check_code: {e}", exc_info=True)
        return jsonify({"result": "An error occurred", "error": str(e)}), 500

def run_static_analysis(user_code):
    logging.debug("Entered run_static_analysis function with user_code:\n%s", user_code)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(user_code.encode('utf-8'))
            temp_file_name = temp_file.name

        result = subprocess.run(
            ['pylint', '--disable=C0114,C0116,C0304', temp_file_name],
            capture_output=True,
            text=True,
            check=False
        )
        filtered_output = parse_pylint_output(result.stdout)
        logging.debug("Completed static analysis with stdout:\n%s", filtered_output)
        logging.error("Static analysis error:\n%s", result.stderr)

        return filtered_output + result.stderr
    except subprocess.CalledProcessError as e:
        logging.error("Static analysis failed with error:\n%s", e.stderr)
        return e.stderr
    finally:
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

def parse_pylint_output(output):
    lines = output.splitlines()
    filtered_lines = []
    for line in lines:
        if line.startswith("*************"):
            continue
        match = re.match(r'.*:\d+:\d+: (.+)', line)
        if match:
            filtered_lines.append(match.group(1))
        else:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)

def extract_initial_explanation(explanation):
    sentences = re.split(r'[.:]\s*', explanation)
    return '. '.join(sentences[:2])

def extract_programming_keywords(text):
    doc = nlp(text)
    keywords = set()

    programming_keywords = {
        "for", "for", "for loop", "while", "while loop", "while",
        "if", "if", "if statement", "def", "def", "function", "functions", "function", "return",
        "else", "elif", "class", "import", "try", "except", "concatenate", "concatenation", "append",
        "add", "addition", "subtract", "subtraction", "multiply", "multiplication", "divide", "division",
        "loop", "loops", "condition", "conditions", "variable", "variables", "list", "dictionary", "set",
        "tuple", "range", "open", "read", "write", "import", "combine", "hash map", "key error", "hash",
        "hash table", "table", "modulus", "arithmetic", "maths", "math", "remainder", "modulo", "divide",
        "division", "integer", "float", "string", "integer division",
    }

    # Extract keywords using spaCy
    for token in doc:
        lemma = token.lemma_
        if lemma in synonyms:
            keywords.update(synonyms[lemma])
        if lemma in programming_keywords:
            keywords.add(lemma)
    
    logging.info(f"Extracted programming keywords: {keywords}")
    return keywords

def get_related_code_by_keywords(keywords, correct_code_examples):
    logging.info(f"Matching keywords: {keywords}")
    
    matching_examples = []
    
    for category, data in correct_code_examples.items():
        logging.debug(f"Processing category: {category} with data: {data}")
        
        if isinstance(data, dict) and "label" in data and "examples" in data:
            for key in data["label"]:
                fuzzy_scores = [(fuzz.partial_ratio(keyword, key), keyword) for keyword in keywords]
                
                for score, keyword in fuzzy_scores:
                    logging.debug(f"Fuzzy matching score between '{keyword}' and '{key}': {score}")
                
                if any(score > 80 for score, keyword in fuzzy_scores):
                    logging.info(f"Fuzzy match found: Label '{key}' matches with keyword '{keyword}' (score: {score})")
                    matching_examples.extend(data["examples"])
                    break
        else:
            logging.warning(f"Unexpected data structure in category '{category}': {data}")
    
    if not matching_examples:
        logging.warning("No related code examples found.")
        return {"incorrect_code": "No related code examples found.", "task_description": "", "description": "", "explanation": "", "task_id": "N/A"}
    
    selected_example = random.choice(matching_examples)
    logging.info(f"Selected example based on keywords: {selected_example}")
    return selected_example

def find_correct_example(task_id, correct_code_examples):
    for data in correct_code_examples.items():
        correct_example = next((ex for ex in data["examples"] if str(ex["task_id"]) == str(task_id)), None)
        if correct_example:
            logging.info(f"Found correct example for task ID {task_id}: {correct_example}")
            return correct_example
    logging.warning(f"No correct example found for task ID {task_id}")
    return None

def format_code_snippets(response):
    code_block_pattern = re.compile(r'```(?:python)?\n(.*?)\n```', re.DOTALL)
    def replace_code_block(match):
        code = match.group(1).strip()
        highlighted_code = highlight(code, PythonLexer(), HtmlFormatter(noclasses=True))
        return f"<pre><code>{highlighted_code}</code></pre>"
    return code_block_pattern.sub(replace_code_block, response)

if __name__ == "__main__":
    app.run(debug=True)