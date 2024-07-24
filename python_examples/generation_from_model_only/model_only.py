import os
import json
import logging
import requests
from typing import Optional

from tqdm import tqdm
from dotenv import load_dotenv

def send_request_to_local_llm(prompt: str, model: str, temperature: float, max_tokens: int):
    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def generate_beginner_questions(model: str, num_questions: int, temperature: float, max_tokens: int) -> list:
    questions = []
    prompt = ("Please generate a simple beginner-level Python programming question. "
            "The question should be clear and straightforward, suitable for someone who is just starting to learn Python. "
            "Focus on basic topics such as loops, variables, data types, control flow, lists, and functions. "
            "Avoid using any meta-text such as 'Here's a beginner-level Python programming question' in your response. "
            "Ensure that the question is straightforward and does not require advanced knowledge of Python.")
    
    for _ in tqdm(range(num_questions), desc="Generating questions"):
        response = send_request_to_local_llm(prompt, model, temperature, max_tokens)
        question = response["choices"][0]["message"]["content"].strip()
        questions.append({"instruction": question})
    
    return questions

def save_generated_questions(questions, output_file_path):
    with open(output_file_path, 'w') as f:
        json.dump(questions, f, indent=4)

def main(output_file_path, num_questions):
    load_dotenv(override=True)
    logging.basicConfig(filename="app.log", filemode="w", format='%(name)s - %(levellevelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    model_name = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
    temperature = 1.0
    max_tokens = 2048  # Adjust as necessary

    print(f"Generating {num_questions} beginner Python questions...")
    questions = generate_beginner_questions(model_name, num_questions, temperature, max_tokens)
    
    save_generated_questions(questions, output_file_path)
    print(f"Generated questions saved to {output_file_path}. Total questions: {len(questions)}")

if __name__ == "__main__":
    output_dir = './python_examples/generation_from_model_only/outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'generated_python_questions.json')
    num_questions = 10  # Change to generate more or fewer questions
    main(output_file_path, num_questions)
