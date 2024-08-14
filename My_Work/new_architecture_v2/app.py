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
from spacy.matcher import PhraseMatcher
from flask_socketio import SocketIO, emit

app = Flask(__name__)

# Set up logging
log_path = os.path.join(os.getcwd(), 'My_Work', 'new_architecture_v2', 'src', 'logs', 'chat_interactions.log')
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

# Load datasets
correct_code_examples = load_dataset('./My_Work/new_architecture_v2/data/code_examples.json')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize PhraseMatcher
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# Add patterns to the matcher
phrases = ["for loop", "while loop", "if statement", "if statements","hash table", "key-value", "conditional statement", "key value pair", "conditional statements"]
patterns = [nlp.make_doc(phrase) for phrase in phrases]
phrase_matcher.add("PROGRAMMING_PHRASES", patterns)

# Synonyms for better keyword matching
synonyms = {
    "append": ["add", "insert", "push"],
    "variable": ["var", "name", "identifier"],
    "list": ["array", "vector"],
    "dictionary": ["dict", "map", "hash map", "hash table", "key-value"],
    "tuple": ["pair", "ordered pair"],
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

    if "python" not in user_question.lower():
        user_question += " (programming Language: python)"
        logging.info(f"Modified user question: {user_question}")
    
    explanation = generate_explanation(user_question, "TheBloke/CodeLlama-13B-Instruct-GGUF")
    logging.info(f"Generated explanation: {explanation}")
    
    initial_explanation = extract_initial_explanation(explanation)
    logging.info(f"Initial part of the explanation: {initial_explanation}")
    
    user_keywords = extract_programming_keywords(user_question)
    explanation_keywords = extract_programming_keywords(initial_explanation)
    combined_keywords = user_keywords.union(explanation_keywords)
    logging.info(f"Extracted user keywords: {user_keywords}")
    logging.info(f"Extracted explanation keywords: {explanation_keywords}")
    logging.info(f"Combined keywords: {combined_keywords}")
    
    incorrect_code_data = get_related_code_by_keywords(combined_keywords, correct_code_examples)
    logging.info(f"Selected incorrect code example: {incorrect_code_data}")
    
    logging.info(f"Task ID: {incorrect_code_data.get('task_id', 'No task ID found')}")
    
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
    
# ORIGINAL CHAT CODE

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message")
    task_id = request.form.get("task_id")
    #task_description = request.form.get("task_description")
    logging.info(f"User chat message: {user_message} for task ID: {task_id}")

    # Retrieve the correct example associated with the task ID
    correct_example = find_correct_example(task_id, correct_code_examples)

    task_description = correct_example.get("task_description", "")
    incorrect_code = correct_example.get("incorrect_code", "")

    # Generate a response using the LLM based on the current task
    explanation = generate_explanation(f"{user_message} (Incorrect Code: {incorrect_code}), (Task Description: {task_description})", "TheBloke/CodeLlama-13B-Instruct-GGUF")
    logging.info(f"Generated chat explanation: {explanation}")
    
    # Format the explanation to ensure code snippets are highlighted
    formatted_explanation = format_code_snippets(explanation)
    logging.info(f"Formatted chat explanation: {formatted_explanation}")

    return jsonify({"response": formatted_explanation})

# CHAT WITH PROMPT - NOT VERY GOOD, BASICALLY BECOMES NOT A CHATBOT AND IS MORE OF A Q&A BOT WITH PROMPT AND RESPONSE.
"""@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message")
    task_id = request.form.get("task_id")
    logging.info(f"User chat message: {user_message} for task ID: {task_id}")

    # Retrieve the correct example associated with the task ID
    correct_example = find_correct_example(task_id, correct_code_examples)
    
    if correct_example:
        task_description = correct_example.get("task_description", "")
        incorrect_code = correct_example.get("incorrect_code", "")
        
        # Refine the prompt with clear context separation
        prompt = (f"Task Description: {task_description}\n"
                  f"Incorrect Code:\n```\n{incorrect_code}\n```\n"
                  "Please help the user understand the task without giving out the solution to the problem:"
                  f" '{user_message}'")

        explanation = generate_explanation(prompt, "TheBloke/CodeLlama-13B-Instruct-GGUF")
        logging.info(f"Generated chat explanation: {explanation}")

        # Format the explanation for consistent code formatting
        formatted_explanation = format_code_snippets(explanation)
        logging.info(f"Formatted chat explanation: {formatted_explanation}")

        return jsonify({"response": formatted_explanation})
    else:
        logging.error(f"No matching task found for task ID: {task_id}")
        return jsonify({"response": "Sorry, I couldn't find any information about this task."})"""

# THIS WORKS BETTER, INCLUDES THE FUNCTION NEEDS_CONTEXT TO DETERMINE IF THE USER'S MESSAGE REQUIRES TASK-SPECIFIC CONTEXT.
"""@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message")
    task_id = request.form.get("task_id")
    logging.info(f"User chat message: {user_message} for task ID: {task_id}")

    # Find the correct example based on the task ID
    correct_example = find_correct_example(task_id, correct_code_examples)
    
    if correct_example:
        task_description = correct_example.get("task_description", "")
        incorrect_code = correct_example.get("incorrect_code", "")

        # Determine if the user's message requires task-specific context
        if needs_context(user_message):
            # Include the task context in the prompt
            prompt = (
                f"Task Description: {task_description}\n"
                f"Incorrect Code:\n```\n{incorrect_code}\n```\n"
                f"User's Question: '{user_message}'\n"
                "Please help the user understand the task without giving out the solution to the problem."
            )
        else:
            # Use the user's message directly without adding task context
            prompt = user_message

        # Generate explanation using the refined prompt
        explanation = generate_explanation(prompt, "TheBloke/CodeLlama-13B-Instruct-GGUF")
        logging.info(f"Generated chat explanation: {explanation}")

        # Format the explanation for display
        formatted_explanation = format_code_snippets(explanation)
        logging.info(f"Formatted chat explanation: {formatted_explanation}")

        return jsonify({"response": formatted_explanation})
    else:
        logging.error(f"No matching task found for task ID: {task_id}")
        return jsonify({"response": "Sorry, I couldn't find any information about this task."})

def needs_context(user_message):
    #Determine if the user's message needs task-specific context.
    # List of keywords that indicate the user is asking for help or clarification
    help_keywords = ["help", "why", "explain", "understand", "what", "how", "problem", "issue"]

    # Check if any of the keywords are in the user's message
    return any(keyword in user_message.lower() for keyword in help_keywords)"""


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

    # Match multi-word phrases like "for loop", "while loop"
    matches = phrase_matcher(doc)
    matched_phrases = [doc[start:end].text.replace(" ", "_") for match_id, start, end in matches]
    keywords.update(matched_phrases)

    # Existing keywords and synonyms logic
    programming_keywords = {
        "def", "function", "return", "functions",
        "else", "elif", "class", "import", "try", "except",
        "concatenate", "append", "add", "subtract", "multiply", "divide",
        "variable", "list", "dictionary", "set",
        "tuple", "range", "open", "read", "write", "combine",
        "hash map", "key error", "hash", "hash table", "modulus",
        "arithmetic", "maths", "remainder", "modulo", "integer",
        "float", "string", "integer division", "while loop", "for loop", 
        "key-value", "key value", "key value pair", "key-value pair",
        "if_statement", "if statement", "for_loop", "while_loop", "hash_table", "key-value",
        "addition", "subtraction", "multiplication", "division","+","-","*","/","%","//",
        "conditions", "conditional statement", "conditional statements",
    }

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
    best_match_score = 0
    best_match_example = None

    for category, data in correct_code_examples.items():
        logging.debug(f"Processing category: {category} with data: {data}")
        
        if isinstance(data, dict) and "label" in data and "examples" in data:
            for key in data["label"]:
                fuzzy_scores = [(fuzz.partial_ratio(keyword, key), keyword) for keyword in keywords]
                
                for score, keyword in fuzzy_scores:
                    logging.debug(f"Fuzzy matching score between '{keyword}' and '{key}': {score}")
                    
                    # Only update if the new score is higher than the current best match
                    if score > best_match_score:
                        best_match_score = score
                        best_match_example = random.choice(data["examples"])

                    # Optionally, add a threshold for matching to avoid irrelevant selections
                    if score > 80:
                        matching_examples.extend(data["examples"])
        
    if not best_match_example:
        logging.warning("No related code examples found.")
        return {"incorrect_code": "No related code examples found.", "task_description": "", "description": "", "explanation": "", "task_id": "N/A"}
    
    logging.info(f"Selected example based on highest match score: {best_match_example}")
    return best_match_example

def find_correct_example(task_id, correct_code_examples):
    for category, data in correct_code_examples.items():
        logging.debug(f"Processing category: {category} with data: {data}")

        if isinstance(data, dict) and "examples" in data:
            examples = data["examples"]

            if isinstance(examples, list):
                correct_example = next((ex for ex in examples if str(ex["task_id"]) == str(task_id)), None)
                
                if correct_example:
                    logging.info(f"Found correct example for task ID {task_id}: {correct_example}")
                    return correct_example
            else:
                logging.warning(f"'examples' in category '{category}' is not a list: {examples}")
        else:
            logging.warning(f"Data in category '{category}' is not a dictionary or lacks 'examples' key: {data}")
    
    logging.error(f"No correct example found for task ID: {task_id}")
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
