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

    # incorrect code prompt v1
    """prompt_answer = ( 
    "Please generate an explanation and incorrect code for the following beginner-level Python programming question. "
    "The answer should include:\n"
    "1. An explanation of the problem, detailed enough for a beginner to understand the fundamental concept behind the question.\n"
    "2. Some INCORRECT or MISSING code based on the question, so the user can try and learn the concepts by debugging, THE CODE MUST INCLUDE SOME KIND OF MISTAKE, THE CODE CANNOT BE CORRECT. \n"
    "3. A prompt asking the user to spot the problem in the code. DO NOT provide the correct code. DO NOT EXPLAIN why the code is incorrect. This is for the user to try and understand.\n"
    "Avoid using any meta-text such as 'Here's the answer' or any phrases indicating the correct solution. DO NOT EXPLAIN WHY THE CODE IS INCORRECT.\n\n"
    "Question:\n{question}\n\n"
    "Answer:\n"
    "Explanation:\n"
    "Provide a clear and concise explanation of the problem.\n\n"
    "Incorrect Code:\n"
    "Provide some incorrect code that a beginner might write when trying to solve the problem. THE CODE MUST INCLUDE SOME KIND OF MISTAKE OR HAVE MISSING PARTS, THE CODE CANNOT BE CORRECT.\n\n"
    "Prompt:\n"
    "Ask the user to identify the problem in the code. DO NOT provide an explanation AS TO WHY THE CODE ISN'T WORKING. DO NOT PROVIDE AN EXPLANATION as to why the code is not working. NOT providing an explanation is very important.\n\n"
    "IMPORTANT: DO NOT EXPLAIN WHY THE CODE IS INCORRECT. ONLY PROVIDE THE INCORRECT CODE AND ASK THE USER TO IDENTIFY THE PROBLEM. THE INCORRECT CODE MUST INCLUDE SOME KIND OF MISTAKE OR HAVE MISSING PARTS THAT MAKE THE CODE INCORRECT."
    )""" #maybe add more missing code here

    # v2
    prompt_answer = (
    "Please generate an explanation and incorrect code for the following beginner-level Python programming question. "
    "The answer should include:\n"
    "1. An explanation of the problem, detailed enough for a beginner to understand the fundamental concept behind the question.\n"
    "2. Some INCORRECT or MISSING code based on the question, so the user can try and learn the concepts by debugging. THE CODE MUST INCLUDE SOME KIND OF MISTAKE, THE CODE CANNOT BE CORRECT. Examples of mistakes include syntax errors, logical errors, incorrect function usage, missing return statements, incorrect variable names. These mistakes should make logical sense for the given question.\n"
    "3. A prompt asking the user to spot the problem in the code. DO NOT provide the correct code. DO NOT EXPLAIN why the code is incorrect. This is for the user to try and understand.\n"
    "Avoid using any meta-text such as 'Here's the answer' or any phrases indicating the correct solution. DO NOT EXPLAIN WHY THE CODE IS INCORRECT.\n\n"
    "Question:\n{question}\n\n"
    "Answer:\n"
    "Explanation:\n"
    "Provide a clear and concise explanation of the problem.\n\n"
    "Incorrect Code:\n"
    "Provide some incorrect code that a beginner might write when trying to solve the problem. THE CODE MUST INCLUDE SOME KIND OF MISTAKE OR HAVE MISSING PARTS, THE CODE CANNOT BE CORRECT. Examples of mistakes include syntax errors, logical errors, incorrect function usage, missing return statements, incorrect variable names, etc.\n\n"
    "Prompt:\n"
    "Ask the user to identify the problem in the code. DO NOT provide an explanation AS TO WHY THE CODE ISN'T WORKING. DO NOT PROVIDE AN EXPLANATION as to why the code is not working. NOT providing an explanation is very important.\n\n"
    "IMPORTANT: DO NOT EXPLAIN WHY THE CODE IS INCORRECT. ONLY PROVIDE THE INCORRECT CODE AND ASK THE USER TO IDENTIFY THE PROBLEM. THE INCORRECT CODE MUST INCLUDE SOME KIND OF MISTAKE OR HAVE MISSING PARTS THAT MAKE THE CODE INCORRECT."
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
    input_file_path = 'python_examples/combined_questions/Datasets/half_of_baoviwnf.json' # just generated from model. didnt seem to change a lot.
    #input_file_path =  './python_examples/combined_questions/combined_output.json' #combined 
    output_dir = './python_examples/answered_questions/outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique file name
    random_string = generate_random_string()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    temperature = "02"
    output_file_path = os.path.join(output_dir, f'paired_python_qa_{timestamp}_{random_string}_temp_{temperature}.json')
    
    sample_size = 10 #None  # Set to None to process the entire dataset
    main(input_file_path, output_file_path, sample_size)