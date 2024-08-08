import os
import random
import logging
import re
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

# Load datasets
correct_code_examples = load_dataset('./My_Work/new_architecture_v2/data/code_examples.json')

# Load spaCy model
import spacy
nlp = spacy.load("en_core_web_sm")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form["question"]
    logging.info(f"User question: {user_question}")
    
    explanation = generate_explanation(user_question, "TheBloke/CodeLlama-13B-Instruct-GGUF")  # was "deepseekcoder"
    correct_code = get_related_code(user_question, correct_code_examples)
    combined_response = create_combined_response(explanation, correct_code).strip()
    
    # Apply syntax highlighting to the code
    combined_response = format_code_snippets(combined_response)
    
    logging.info(f"Generated response: {combined_response}")
    
    return render_template("index.html", question=user_question, response=combined_response)

def get_related_code(question, correct_code_examples):
    doc = nlp(question)
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    logging.info(f"Extracted keywords: {keywords}")

    best_match = None
    highest_score = 0

    for example in correct_code_examples:
        hint_doc = nlp(example["hint"])
        hint_keywords = [token.lemma_ for token in hint_doc if token.is_alpha and not token.is_stop]
        score = len(set(keywords) & set(hint_keywords))
        if score > highest_score:
            highest_score = score
            best_match = example

    return best_match["code"] if best_match else random.choice(correct_code_examples)["code"]

def format_code_snippets(response):
    # This regex finds all code blocks wrapped in triple backticks
    code_block_pattern = re.compile(r'```(python)?\n(.*?)\n```', re.DOTALL)
    
    def replace_code_block(match):
        code = match.group(2).strip()
        highlighted_code = highlight(code, PythonLexer(), HtmlFormatter())
        return f"<div class='code-block'>{highlighted_code}</div>"

    formatted_response = code_block_pattern.sub(replace_code_block, response)
    return formatted_response

if __name__ == "__main__":
    app.run(debug=True)
