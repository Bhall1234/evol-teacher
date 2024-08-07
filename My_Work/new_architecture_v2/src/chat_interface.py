# Contains the user interface for interacting with the chatbot.
import json
import random
import time
from src.explanation_generation import generate_explanation
from src.response_combination import create_combined_response
from src.utils import save_log

def get_related_code(question, correct_code_examples):
    # Simple matching based on keywords. You can expand this to be more sophisticated.
    for example in correct_code_examples:
        if "for loop" in question.lower() and "for loop" in example["hint"].lower():
            return example["code"]
        elif "while loop" in question.lower() and "while loop" in example["hint"].lower():
            return example["code"]
        # Add more conditions as needed for other types of questions
    return random.choice(correct_code_examples)["code"]

def chat_interface(correct_code_examples, user_questions):
    log_file_path = './My_Work/new_architecture_v2/src/logs/chat_interactions.log'
    while True:
        user_question = input("Ask a question about Python: ")
        print("Processing your question...")
        explanation = generate_explanation(user_question, "TheBloke/CodeLlama-13B-Instruct-GGUF") # was "deepseekcoder"
        
        correct_code = get_related_code(user_question, correct_code_examples)
        
        combined_response = create_combined_response(user_question, explanation, correct_code)
        
        # Mimic chatbot response by gradually displaying the text
        for line in combined_response.split('\n'):
            print(line)
            time.sleep(0.5)

        # Log the interaction
        log_data = {
            "user_question": user_question,
            "explanation": explanation,
            "incorrect_code": correct_code
        }
        save_log(log_data, log_file_path)

if __name__ == "__main__":
    chat_interface()
