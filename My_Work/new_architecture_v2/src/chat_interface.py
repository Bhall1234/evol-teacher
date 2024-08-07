# Contains the user interface for interacting with the chatbot.
from src.explanation_generation import generate_explanation
from src.response_combination import create_combined_response
import random
import time

def chat_interface(correct_code_examples, user_questions):
    while True:
        user_question = input("Ask a question about Python: ")
        explanation = generate_explanation(user_question, "TheBloke/CodeLlama-13B-Instruct-GGUF") # was "deepseekcoder"
        correct_code = random.choice(correct_code_examples)["code"]
        combined_response = create_combined_response(user_question, explanation, correct_code)
        
        # Mimic chatbot response by gradually displaying the text
        for line in combined_response.split('\n'):
            print(line)
            time.sleep(0.5)

if __name__ == "__main__":
    chat_interface()
