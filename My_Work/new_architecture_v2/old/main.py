import os
import time
from src.utils import load_dataset
from src.chat_interface import chat_interface

def main():
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # Load datasets
    correct_code_examples = load_dataset('./My_Work/new_architecture_v2/data/code_examples.json')
    user_questions = load_dataset('./My_Work/new_architecture_v2/data/user_questions.json')

    # Initialize chat interface
    chat_interface(correct_code_examples, user_questions)

if __name__ == "__main__":
    main()
