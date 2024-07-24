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

def save_filtered_dataset(filtered_data, output_file_path):
    with open(output_file_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)

def filter_python_examples(dataset):
    filtered_data = []
    for item in dataset:
        instruction = item.get('instruction', '').lower()
        output = item.get('output', '').lower()
        if 'python' in instruction or 'python' in output:
            filtered_data.append(item)
    return filtered_data

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

def diversify_questions(instructions, model: str, temperature: float, max_tokens: int) -> list:
    methods = [
        'Rephrase the instruction.',
        'Change variable names.',
        'Add a small real-world context to the question.',
        'Alter the sample data provided in the question.',
        'Combine two related tasks into one question.'
    ]
    
    new_tasks = []
    for task in instructions:
        chosen_method = random.choice(methods)
        prompt = f"Please diversify the given programming test question using the following method:\n{chosen_method}\n\n#Given Test#\n{task['instruction']}\n\n#Rewritten Test#\n"
        response = send_request_to_local_llm(prompt, model, temperature, max_tokens)
        diversified_response = response["choices"][0]["message"]["content"]
        new_tasks.append({
            "instruction": diversified_response,
            "output": task['output']
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
    output_dir="./new_generation/",
    seed_tasks_path="./formatted_dataset.json",
    evolutions=3,
    temperature=1,
    max_tokens=2048,
    frequency_penalty=0,
    top_p=0.9,
    model_name="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    num_instructions=343
):
    load_dotenv(override=True)
    logging.basicConfig(filename="app.log", filemode="w", format='%(name)s - %(levellevelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    prev_tasks = load_dataset(seed_tasks_path)
    start_time = time.time()

    for evolution in range(1, evolutions + 1):
        print(f'Evolution {evolution}:')
        evolution_start_time = time.time()

        # Generate diverse responses
        print("Generating Diverse Responses")
        new_tasks = diversify_questions(prev_tasks, model_name, temperature, max_tokens)
        new_tasks = [task for task in new_tasks if not check_instruction(task)]

        # Print some of the new tasks for verification
        for task in new_tasks[:5]:
            print(json.dumps(task, indent=2))

        # Output to a JSON file with pretty printing
        output_file = os.path.join(output_dir, f"diverse_responses_evolution_{evolution}.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w") as json_file:
            json.dump(new_tasks, json_file, indent=4)

        prev_tasks = new_tasks
        evolution_time = time.time() - evolution_start_time
        print(f'Evolution {evolution} complete, took {evolution_time:.2f}s')

    final_time = time.time() - start_time
    print(f'All Computation complete, total run took {final_time:.2f}s')

if __name__ == "__main__":
    generate_diverse_dataset()
