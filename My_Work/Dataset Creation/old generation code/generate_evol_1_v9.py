import os
import json
import glob
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


def load_instructions(file_path: str, num_instructions: int = 10):
    with open(file_path, "r") as json_file:
        loaded_json = json.load(json_file)
    return loaded_json[:num_instructions]


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


def generate_incorrect_responses(instructions, model: str, temperature: float, max_tokens: int) -> list:
    methods = [
        'Add an off-by-one error in the loop range.',
        'Remove or misplace indentation in the code.',
        'Use an incorrect operator in the condition.',
        'Omit a return statement in the function.',
        'Replace a commonly used function with a non-existent one.'
    ]
    
    response_templates = [
        "I've written a piece of code based on your instruction, but I've included a small mistake on purpose. Can you figure out what's wrong?\n\n{}",
        "Here's the code you asked for, but be aware that I've intentionally added an error. Can you spot and fix it?\n\n{}",
        "I completed your instruction with a purposeful mistake. See if you can identify the error and correct it.\n\n{}",
        "Take a look at this code. It includes a deliberate mistake based on your instruction. Can you find and fix it?\n\n{}",
        "The following code has been written with your instruction in mind, but it contains an intentional error. Can you spot and correct it?\n\n{}"
    ]

    new_tasks = []
    for task in instructions:
        chosen_method = random.choice(methods)
        chosen_template = random.choice(response_templates)
        prompt = f"Generate a code response for the following instruction with an intentional mistake using the method '{chosen_method}'.\n\nInstruction: {task['instruction']}\n\nResponse with Mistake:"
        response = send_request_to_local_llm(prompt, model, temperature, max_tokens)
        incorrect_response = response["choices"][0]["message"]["content"]
        new_tasks.append({
            "instruction": task['instruction'],
            "output": chosen_template.format(incorrect_response)
        })
    return new_tasks


def check_instruction(instruction) -> bool:
    content = instruction["output"]
    if not content:
        return True
    if len(content.split()) <= 3:
        return True
    if not content[0].isascii():
        return True
    return False


def generate_incorrect_dataset(
    output_dir="./new_generation/",
    seed_tasks_path="./generation/EvolInstruct-Code-8k.json",
    temperature=1,
    max_tokens=2048,
    frequency_penalty=0,
    top_p=0.9,
    model_name="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    num_instructions=10  # Number of instructions to process for testing
):
    load_dotenv(override=True)
    logging.basicConfig(filename="app.log", filemode="w", format='%(name)s - %(levellevelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    prev_tasks = load_instructions(seed_tasks_path, num_instructions)
    start_time = time.time()

    # Generate incorrect responses
    print("Generating Incorrect Responses")
    new_tasks = generate_incorrect_responses(prev_tasks, model_name, temperature, max_tokens)
    new_tasks = [task for task in new_tasks if not check_instruction(task)]

    # Print some of the new tasks for verification
    for task in new_tasks[:5]:
        print(json.dumps(task, indent=2))

    # Output to a JSON file
    output_file = output_dir + "incorrect_responses.json"
    with open(output_file, "w") as json_file:
        json.dump(new_tasks, json_file)

    final_time = time.time() - start_time
    print(f'All Computation complete, total run took {final_time:.2f}s')


if __name__ == "__main__":
    generate_incorrect_dataset()