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

    prompt_answer = (
    "Please remove any explanation of the incorrect code snippet that you can see and provide the output without this explanation. DO NOT EDIT THE REST OF THE OUTPUT."
    "Question:\n{question}\n\n"
    "Answer:\n"
    "Explanation:\n"
    "Provide a clear and concise explanation of the problem.\n\n"
    "Incorrect Code:\n"
    "Provide some incorrect code that a beginner might write when trying to solve the problem.\n\n"
    "Prompt:\n"
    "Ask the user to identify the problem in the code. Remove any explanation of the broken/incorrect code, this is so the user has a chance to understand the problem."
)

    for question in tqdm(questions, desc="Generating answers"):
        prompt = prompt_answer.format(question=question['instruction'])
        response = send_request_to_local_llm(prompt, model, temperature, max_tokens)
        answer = response["choices"][0]["message"]["content"].strip()
        paired_qa.append({"instruction": question['instruction'], "output": answer}) #changed answer to output
    
    return paired_qa

def save_paired_qa(paired_qa, output_file_path):
    with open(output_file_path, 'w') as f:
        json.dump(paired_qa, f, indent=4)

def generate_random_string(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def main(input_file_path, output_file_path, sample_size=None):
    load_dotenv(override=True)
    logging.basicConfig(filename="app.log", filemode="w", format='%(name)s - %(levellevelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    model_name = "TheBloke/CodeLlama-13B-Instruct-GGUF"
    temperature = 0.2 # was 0.8
    max_tokens = 16384  # Adjust as necessary 2048

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
    input_file_path = './python_examples/answered_questions/outputs/paired_python_qa_2024-07-29_12-34-06_yrpxmede.json'
    output_dir = './python_examples/answered_questions/outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique file name
    random_string = generate_random_string()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file_path = os.path.join(output_dir, f'paired_python_qa_post_process{timestamp}_{random_string}.json')
    
    sample_size = 10 #None  # Set to None to process the entire dataset
    main(input_file_path, output_file_path, sample_size)