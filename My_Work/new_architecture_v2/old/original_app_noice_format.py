#ORGINAL
"""import os
import random
import logging
from flask import Flask, request, render_template
from src.utils import load_dataset
from src.explanation_generation import generate_explanation
from src.response_combination import create_combined_response

app = Flask(__name__)

# Set up logging
log_path = os.path.join(os.getcwd(), 'My_Work', 'new_architecture_v2', 'src', 'logs', 'chat_interactions.log')
os.makedirs(os.path.dirname(log_path), exist_ok=True)  # Ensure the log directory exists
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

# Load datasets
correct_code_examples = load_dataset('./My_Work/new_architecture_v2/data/code_examples.json')
user_questions = load_dataset('./My_Work/new_architecture_v2/data/user_questions.json')

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
    
    logging.info(f"Generated response: {combined_response}")
    
    return render_template("index.html", question=user_question, response=combined_response)

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

if __name__ == "__main__":
    app.run(debug=True)"""