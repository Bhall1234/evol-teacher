import os
import json
import logging
import requests
from typing import Optional
import random
import string
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
import time

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

def load_questions(file_path: str):
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_answers(questions, model: str, temperature: float, max_tokens: int) -> list:
    paired_qa = []
    prompt_answer = ("Please generate an explanation and an incorrect code for the following beginner-level Python programming question. "
                     "The answer should include:\n"
                     "1. An explanation of the problem.\n"
                     "2. Some incorrect code based on the question.\n"
                     "3. A prompt asking the user to spot the problem in the code.\n"
                     "Avoid using any meta-text such as 'Here's the answer' in your response.\n\n"
                     "Question:\n{question}\n\n"
                     "Answer:")

    for question in tqdm(questions, desc="Generating answers"):
        prompt = prompt_answer.format(question=question['instruction'])
        response = send_request_to_local_llm(prompt, model, temperature, max_tokens)
        answer = response["choices"][0]["message"]["content"].strip()
        paired_qa.append({"instruction": question['instruction'], "answer": answer})
    
    return paired_qa

def save_paired_qa(paired_qa, output_file_path):
    with open(output_file_path, 'w') as f:
        json.dump(paired_qa, f, indent=4)

def generate_random_string(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def main(input_file_path, output_file_path, sample_size):
    load_dotenv(override=True)
    logging.basicConfig(filename="app.log", filemode="w", format='%(name)s - %(levellevelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    model_name = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
    temperature = 1.0
    max_tokens = 2048  # Adjust as necessary

    start_time = time.time()
    print(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")

    questions = load_questions(input_file_path)
    
    if sample_size and sample_size < len(questions):
        questions = questions[:sample_size]
    
    print(f"Generating answers for {len(questions)} beginner Python questions...")
    paired_qa = generate_answers(questions, model_name, temperature, max_tokens)
    
    save_paired_qa(paired_qa, output_file_path)
    print(f"Paired questions and answers saved to {output_file_path}. Total pairs: {len(paired_qa)}")

    end_time = time.time()
    print(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    #input_file_path = './python_examples/evolved_questions/outputs/evolved_python_questions_20240724173137_bkhpcvgx.json' # EVOLVED
    input_file_path = './python_examples/generation_from_model_only/outputs/generated_python_questions_20240724171958_weptfvyg.json' # NOT EVOLVED
    output_dir = './python_examples/answered_questions/outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique file name
    random_string = generate_random_string()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file_path = os.path.join(output_dir, f'paired_python_qa_{timestamp}_{random_string}.json')
    
    sample_size = 5  # Change to limit the number of questions processed for testing
    main(input_file_path, output_file_path, sample_size)