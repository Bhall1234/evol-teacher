import os
import random
import logging
from flask import Flask, request, render_template
from src.utils import load_dataset
from src.explanation_generation import generate_explanation
from src.response_combination import create_combined_response
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

app = Flask(__name__)

# Set up logging
log_path = os.path.join(os.getcwd(), 'My_Work', 'new_architecture_v2', 'src', 'logs', 'chat_interactions.log')
os.makedirs(os.path.dirname(log_path), exist_ok=True)  # Ensure the log directory exists
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

# Load datasets, user_questions inst necessary right now, could be good for testing later on.
correct_code_examples = load_dataset('./My_Work/new_architecture_v2/data/code_examples.json')
#user_questions = load_dataset('./My_Work/new_architecture_v2/data/user_questions.json')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form["question"]
    logging.info(f"User question: {user_question}")
    
    explanation = generate_explanation(user_question, "TheBloke/CodeLlama-13B-Instruct-GGUF")  # was "deepseekcoder"
    correct_code = get_related_code(user_question, correct_code_examples)
    combined_response = create_combined_response(user_question, explanation, correct_code)
    
    # Apply syntax highlighting to the code
    combined_response = format_code_snippets(combined_response)
    
    logging.info(f"Generated response: {combined_response}")
    
    return render_template("index.html", question=user_question, response=combined_response)

# this is awful, needs to be refactored and improved upon in the future
def get_related_code(question, correct_code_examples):
    if "for loop" in question.lower():
        related_hints = [example for example in correct_code_examples if "for loop" in example["hint"].lower()]
    elif "while loop" in question.lower():
        related_hints = [example for example in correct_code_examples if "while loop" in example["hint"].lower()]
    else:
        related_hints = correct_code_examples

    if related_hints:
        return random.choice(related_hints)["code"]
    else:
        return random.choice(correct_code_examples)["code"]

def format_code_snippets(response):
    parts = response.split("```python")
    formatted_response = parts[0]
    for part in parts[1:]:
        if "```" in part:
            code, rest = part.split("```", 1)
            highlighted_code = highlight(code, PythonLexer(), HtmlFormatter())
            formatted_response += f"<div class='code-block'>{highlighted_code}</div>{rest}"
        else:
            formatted_response += part
    return formatted_response

if __name__ == "__main__":
    app.run(debug=True)
