# IM NOT SURE IF THIS IS ACTUALLY THE BEST WAY OF DOING THIS BUT IT DOES SEEM TO WORK SO FAR. 
# IM COMBINING THE USER QUERY AND THE EXPLANATION TO FIND THE KEYWORDS AND THEN USING THESE KEYWORDS TO FIND RELATED CODE EXAMPLES.

"""import pylint.lint
import tempfile
import io
from contextlib import redirect_stdout
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
    logging.info(f"Initial part of the explanation: {initial_explanation}")
    
    # Extract keywords from the user question and initial explanation
    user_keywords = extract_programming_keywords(user_question)
    explanation_keywords = extract_programming_keywords(initial_explanation)
    
    # Combine the keywords
    combined_keywords = user_keywords.union(explanation_keywords)
    logging.info(f"Combined keywords: {combined_keywords}")
    
    # Match the combined keywords with incorrect code examples
    incorrect_code_data = get_related_code_by_keywords(combined_keywords, correct_code_examples)
    
    # Log the task_id to ensure it's being set correctly
    logging.info(f"Task ID: {incorrect_code_data.get('task_id', 'No task_id found')}")
    
    # Apply syntax highlighting to the explanation and incorrect code
    formatted_explanation = format_code_snippets(explanation)  # Use the full explanation for display
    formatted_incorrect_code = highlight(incorrect_code_data.get("incorrect_code", "No incorrect code found."), PythonLexer(), HtmlFormatter(noclasses=True))
    formatted_correct_code = highlight(incorrect_code_data.get("correct_code", "No correct code found."), PythonLexer(), HtmlFormatter(noclasses=True))
    
    logging.info(f"Generated explanation: {explanation}")
    logging.info(f"Incorrect code: {incorrect_code_data.get('incorrect_code', 'No incorrect code found.')}")
    
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
    try:
        # Use the Python interpreter from the virtual environment
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = e.stderr
    return jsonify({"output": output})

@app.route("/check_code", methods=["POST"])
def check_code():
    try:
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
        
        expected_output = correct_example["expected_output"]
        
        try:
            # Use the Python interpreter from the virtual environment
            result = subprocess.run([sys.executable, "-c", user_code], capture_output=True, text=True, check=True)
            user_output = result.stdout
        except subprocess.CalledProcessError as e:
            user_output = e.stderr
        
        # Compare the user's code output with the expected output
        if user_output.strip() == expected_output.strip():
            result = "Correct"
        else:
            result = "Incorrect"
        
        # Run static analysis
        logging.debug("Running static analysis...")
        static_analysis_result = run_static_analysis(user_code)
        logging.debug("Static analysis result: %s", static_analysis_result)
        

        return jsonify({
            "result": result,
            "static_analysis": static_analysis_result,
        })
    except Exception as e:
        logging.error(f"Error in check_code: {e}", exc_info=True)
        return jsonify({"result": "An error occurred", "error": str(e)}), 500

def run_static_analysis(user_code):
    logging.debug("Entered run_static_analysis function with user_code: %s", user_code)
    try:
        # Create a temporary file with the user code
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(user_code.encode('utf-8'))
            temp_file_name = temp_file.name

        # Run pylint on the temporary file
        result = subprocess.run(
            ['pylint', '--disable=C0114,C0116,C0304', temp_file_name],  # Disable specific pylint warnings that arent relevant to the user.
            capture_output=True,
            text=True,
            check=False  # Don't raise an exception on non-zero exit.
        )

        # Parse output to remove file paths and line numbers
        filtered_output = parse_pylint_output(result.stdout)

        logging.debug("Completed static analysis with stdout: %s", result.stdout)
        logging.error("Static analysis error: %s", result.stderr)

        return filtered_output + result.stderr
    except subprocess.CalledProcessError as e:
        logging.error("Static analysis failed with error: %s", e.stderr)
        return e.stderr
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

def parse_pylint_output(output):
    # Example function to filter and group pylint output
    lines = output.splitlines()
    filtered_lines = []
    for line in lines:
        if line.startswith("*************"):
            continue
        # Remove file path and line numbers
        match = re.match(r'.*:\d+:\d+: (.+)', line)
        if match:
            filtered_lines.append(match.group(1))
        else:
            filtered_lines.append(line)
    
    # Join filtered lines and enhance readability
    return "\n".join(filtered_lines)

def extract_initial_explanation(explanation):
    # Split the explanation into sentences using both '.' and ':', these are the most common sentence delimiters in the explanations.
    sentences = re.split(r'[.:]\s*', explanation)
    initial_part = '. '.join(sentences[:2])  # Adjust the number of sentences as needed
    return initial_part

def extract_programming_keywords(text):
    doc = nlp(text)
    
    # Define a list of programming keywords
    programming_keywords = {    "for","`for`","for loop",
                                "while","while loop","`while`",
                                "`if`","if","if statement",
                                "`def`", "def", "function", "functions", "`function`", "return",
                                "else", "elif","class", "import", "try", "except",
                                "concatenate", "concatenation", "append", "add", "addition",
                                "subtract", "subtraction", "multiply", "multiplication",
                                "divide", "division", "loop", "loops", "condition", "conditions",
                                "variable", "variables", "list", "dictionary", "set", "tuple",
                                "range", "open", "read", "write", "import", "from", "as", "with",
                                "combine", "hash map", "key error", "hash", "hash table", "table"}
    
    # Extract keywords using spaCy and filter to include only programming keywords
    keywords = {token.lemma_ for token in doc if token.lemma_ in programming_keywords}
    
    logging.info(f"Extracted programming keywords from text: {keywords}")
    return keywords

def get_related_code_by_keywords(keywords, correct_code_examples):
    logging.info(f"Matching keywords: {keywords}")
    
    # Filter examples by label
    filtered_examples = [ex for ex in correct_code_examples["examples"] if any(label in ex["label"] for label in keywords)]

    if not filtered_examples:
        return {"incorrect_code": "No related code examples found.", "task_description": "", "description": "", "explanation": "", "task_id": "N/A"}

    best_match = random.choice(filtered_examples)
    return best_match

#def format_code_snippets(response):
    # This regex finds all code blocks wrapped in triple backticks
    #code_block_pattern = re.compile(r'```(python)?\n(.*?)\n```', re.DOTALL)
    
    #def replace_code_block(match):
        # Capture the code block and remove the 'python' if present
        #code = match.group(2).strip()
        #highlighted_code = highlight(code, PythonLexer(), HtmlFormatter(noclasses=True))
        #return f"<pre><code>{highlighted_code}</code></pre>"

    #formatted_response = code_block_pattern.sub(replace_code_block, response)
    #return formatted_response

def format_code_snippets(response):
    # This regex finds all code blocks wrapped in triple backticks
    code_block_pattern = re.compile(r'```(?:python)?\n(.*?)\n```', re.DOTALL)
    
    def replace_code_block(match):
        # Capture the code block content
        code = match.group(1).strip()  # The code content without 'python'
        
        # Use Pygments to highlight the code
        highlighted_code = highlight(code, PythonLexer(), HtmlFormatter(noclasses=True))
        
        # Return the highlighted code wrapped in pre and code tags
        return f"<pre><code>{highlighted_code}</code></pre>"

    # Substituting code blocks in the response with formatted HTML
    formatted_response = code_block_pattern.sub(replace_code_block, response)
    
    return formatted_response


if __name__ == "__main__":
    app.run(debug=True)"""




# NEW JSON FORMAT TESTING
import pylint.lint
import tempfile
import io
from contextlib import redirect_stdout
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
    logging.info(f"Initial part of the explanation: {initial_explanation}")
    
    # Extract keywords from the user question and initial explanation
    user_keywords = extract_programming_keywords(user_question)
    explanation_keywords = extract_programming_keywords(initial_explanation)
    
    # Combine the keywords
    combined_keywords = user_keywords.union(explanation_keywords)
    logging.info(f"Combined keywords: {combined_keywords}")
    
    # Match the combined keywords with incorrect code examples
    incorrect_code_data = get_related_code_by_keywords(combined_keywords, correct_code_examples)
    
    # Log the task_id to ensure it's being set correctly
    logging.info(f"Task ID: {incorrect_code_data.get('task_id', 'No task_id found')}")
    
    # Apply syntax highlighting to the explanation and incorrect code
    formatted_explanation = format_code_snippets(explanation)  # Use the full explanation for display
    formatted_incorrect_code = highlight(incorrect_code_data.get("incorrect_code", "No incorrect code found."), PythonLexer(), HtmlFormatter(noclasses=True))
    formatted_correct_code = highlight(incorrect_code_data.get("correct_code", "No correct code found."), PythonLexer(), HtmlFormatter(noclasses=True))
    
    logging.info(f"Generated explanation: {explanation}")
    logging.info(f"Incorrect code: {incorrect_code_data.get('incorrect_code', 'No incorrect code found.')}")
    
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
    try:
        # Use the Python interpreter from the virtual environment
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = e.stderr
    return jsonify({"output": output})

@app.route("/check_code", methods=["POST"])
def check_code():
    try:
        data = request.get_json()
        user_code = data["code"]
        task_id = data["task_id"]
        
        # Log the task_id to ensure it's being received correctly
        logging.info(f"Received task ID: {task_id} (type: {type(task_id)})")
        
        # Find the correct code example based on the task ID
        correct_example = None
        
        # Iterate through each category in the correct_code_examples
        for category, data in correct_code_examples.items():
            # Search through examples in the current category
            correct_example = next((ex for ex in data["examples"] if str(ex["task_id"]) == str(task_id)), None)
            if correct_example:
                break  # If found, exit the loop

        if not correct_example:
            logging.error("No matching task found.")
            return jsonify({"result": "No matching task found."}), 404
        
        expected_output = correct_example["expected_output"]
        
        try:
            # Use the Python interpreter from the virtual environment
            result = subprocess.run([sys.executable, "-c", user_code], capture_output=True, text=True, check=True)
            user_output = result.stdout
        except subprocess.CalledProcessError as e:
            user_output = e.stderr
        
        # Compare the user's code output with the expected output
        if user_output.strip() == expected_output.strip():
            result = "Correct"
        else:
            result = "Incorrect"
        
        # Run static analysis
        logging.debug("Running static analysis...")
        static_analysis_result = run_static_analysis(user_code)
        logging.debug("Static analysis result: %s", static_analysis_result)
        
        return jsonify({
            "result": result,
            "static_analysis": static_analysis_result,
        })
    except Exception as e:
        logging.error(f"Error in check_code: {e}", exc_info=True)
        return jsonify({"result": "An error occurred", "error": str(e)}), 500

def run_static_analysis(user_code):
    logging.debug("Entered run_static_analysis function with user_code: %s", user_code)
    try:
        # Create a temporary file with the user code
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(user_code.encode('utf-8'))
            temp_file_name = temp_file.name

        # Run pylint on the temporary file
        result = subprocess.run(
            ['pylint', '--disable=C0114,C0116,C0304', temp_file_name],  # Disable specific pylint warnings that arent relevant to the user.
            capture_output=True,
            text=True,
            check=False  # Don't raise an exception on non-zero exit.
        )

        # Parse output to remove file paths and line numbers
        filtered_output = parse_pylint_output(result.stdout)

        logging.debug("Completed static analysis with stdout: %s", result.stdout)
        logging.error("Static analysis error: %s", result.stderr)

        return filtered_output + result.stderr
    except subprocess.CalledProcessError as e:
        logging.error("Static analysis failed with error: %s", e.stderr)
        return e.stderr
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

def parse_pylint_output(output):
    # Example function to filter and group pylint output
    lines = output.splitlines()
    filtered_lines = []
    for line in lines:
        if line.startswith("*************"):
            continue
        # Remove file path and line numbers
        match = re.match(r'.*:\d+:\d+: (.+)', line)
        if match:
            filtered_lines.append(match.group(1))
        else:
            filtered_lines.append(line)
    
    # Join filtered lines and enhance readability
    return "\n".join(filtered_lines)

def extract_initial_explanation(explanation):
    # Split the explanation into sentences using both '.' and ':', these are the most common sentence delimiters in the explanations.
    sentences = re.split(r'[.:]\s*', explanation)
    initial_part = '. '.join(sentences[:2])  # Adjust the number of sentences as needed
    return initial_part

def extract_programming_keywords(text):
    doc = nlp(text)
    
    # Define a list of programming keywords
    programming_keywords = {    "for","`for`","for loop",
                                "while","while loop","`while`",
                                "`if`","if","if statement",
                                "`def`", "def", "function", "functions", "`function`", "return",
                                "else", "elif","class", "import", "try", "except",
                                "concatenate", "concatenation", "append", "add", "addition",
                                "subtract", "subtraction", "multiply", "multiplication",
                                "divide", "division", "loop", "loops", "condition", "conditions",
                                "variable", "variables", "list", "dictionary", "set", "tuple",
                                "range", "open", "read", "write", "import",
                                "combine", "hash map", "key error", "hash", "hash table", "table", "modulus"}
    
    # Extract keywords using spaCy and filter to include only programming keywords
    keywords = {token.lemma_ for token in doc if token.lemma_ in programming_keywords}
    
    logging.info(f"Extracted programming keywords from text: {keywords}")
    return keywords

def get_related_code_by_keywords(keywords, correct_code_examples):
    logging.info(f"Matching keywords: {keywords}")
    
    matching_examples = []

    # Iterate through each category in the correct_code_examples
    for category, data in correct_code_examples.items():
        # Check if any of the keywords match the labels of the current category
        if any(keyword in data["label"] for keyword in keywords):
            # If there's a match, add all examples from this category
            matching_examples.extend(data["examples"])
    
    if not matching_examples:
        return {"incorrect_code": "No related code examples found.", "task_description": "", "description": "", "explanation": "", "task_id": "N/A"}

    # Randomly select one of the matching examples
    selected_example = random.choice(matching_examples)
    
    return selected_example

def format_code_snippets(response):
    # This regex finds all code blocks wrapped in triple backticks
    code_block_pattern = re.compile(r'```(?:python)?\n(.*?)\n```', re.DOTALL)
    
    def replace_code_block(match):
        # Capture the code block content
        code = match.group(1).strip()  # The code content without 'python'
        
        # Use Pygments to highlight the code
        highlighted_code = highlight(code, PythonLexer(), HtmlFormatter(noclasses=True))
        
        # Return the highlighted code wrapped in pre and code tags
        return f"<pre><code>{highlighted_code}</code></pre>"

    # Substituting code blocks in the response with formatted HTML
    formatted_response = code_block_pattern.sub(replace_code_block, response)
    
    return formatted_response

if __name__ == "__main__":
    app.run(debug=True)
