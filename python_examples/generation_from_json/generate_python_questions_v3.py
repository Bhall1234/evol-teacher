import os
import json
import logging
import dataclasses
import random
import time
import requests
from typing import Optional, Sequence

from tqdm import tqdm
from dotenv import load_dotenv

@dataclasses.dataclass
class DecodingArguments:
    max_tokens: int = 2048
    temperature: float = 1
    top_p: float = 0.9
    n: int = 1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = False
    stop: Optional[Sequence[str]] = None

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_dataset(data, output_file_path):
    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=4)

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

def diversify_questions(task, model: str, temperature: float, max_tokens: int) -> list:
    methods = [
        'Rephrase the instruction to make it more concise.',
        'Rewrite the instruction using different wording but keeping the same meaning.',
        'Change the phrasing of the instruction without altering its original intent.'
    ]
    
    new_tasks = []
    for _ in range(5):  # Generate 5 variations for each question
        chosen_method = random.choice(methods)
        prompt = (f"Please rewrite the following programming question using the following method:\n{chosen_method}\n\n"
                  f"#Original Test#\n{task['instruction']}\n\n"
                  "Ensure the rewritten test is clear and simple. "
                  "Do not include any answer, solution, or explanation, just the instruction."
                  "\n\n#Rewritten Instruction#")
        
        response = send_request_to_local_llm(prompt, model, temperature, max_tokens)
        rewritten_instruction = response["choices"][0]["message"]["content"].strip()
        
        # Clean the rewritten instruction from any meta-text
        rewritten_instruction = rewritten_instruction.replace("#Rewritten Instruction#", "").strip()
        
        new_tasks.append({
            "instruction": rewritten_instruction,
            "output": task['output']  # Pair with the original output
        })
    return new_tasks

def check_instruction(instruction) -> bool:
    content = instruction["instruction"]
    if not content:
        return True
    if len(content.split()) <= 3:
        return True
    if not content[0].isascii():
        return True
    return False

def generate_diverse_dataset(
    output_dir="./python_examples/",
    seed_tasks_path="./python_examples/extract_python_only/curated_python_examples.json",
    evolutions=3,
    temperature=1,
    max_tokens=2048,
    frequency_penalty=0,
    top_p=0.9,
    model_name="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    num_instructions=343,
    sample_size=343  # Sample size for testing
):
    load_dotenv(override=True)
    logging.basicConfig(filename="app.log", filemode="w", format='%(name)s - %(levellevelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    prev_tasks = load_dataset(seed_tasks_path)[:sample_size]
    start_time = time.time()

    all_tasks = []

    for evolution in range(1, evolutions + 1):
        print(f'Evolution {evolution}:')
        evolution_start_time = time.time()

        # Generate diverse responses
        print("Generating Diverse Responses")
        for task in tqdm(prev_tasks, desc="Processing tasks"):
            new_tasks = diversify_questions(task, model_name, temperature, max_tokens)
            new_tasks = [task for task in new_tasks if not check_instruction(task)]
            all_tasks.extend(new_tasks)

        # Output to a JSON file with pretty printing
        output_file = os.path.join(output_dir, f"diverse_responses_evolution_{evolution}.json")
        os.makedirs(output_dir, exist_ok=True)
        save_dataset(all_tasks, output_file)

        evolution_time = time.time() - evolution_start_time
        print(f'Evolution {evolution} complete, took {evolution_time:.2f}s')

    final_time = time.time() - start_time
    print(f'All Computation complete, total run took {final_time:.2f}s')

if __name__ == "__main__":
    generate_diverse_dataset()
