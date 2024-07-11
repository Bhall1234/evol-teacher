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


def convert_alpaca_to_evol(file_path: str, lines: bool = False, output_file: str = "converted_alpaca.json"):
    result = []
    if lines:
        with open(file_path, "r") as json_file:
            loaded_json = [json.loads(line) for line in json_file]
        for record in loaded_json:
            if record["instances"][0]["input"]:
                record["instruction"] += '\n' + record["instances"][0]["input"]
            result.append({
                "instruction": record["instruction"],
                "output": record["instances"][0]["output"]
            })
    else:
        with open(file_path, "r") as json_file:
            loaded_json = json.load(json_file)
        for record in loaded_json:
            if record["input"]:
                record["instruction"] += '\n' + record["input"]
            result.append({
                "instruction": record["instruction"],
                "output": record["output"]
            })
    with open(output_file, "w") as fp:
        json.dump(result, fp)
    return result


def merge_evolutions(output_dir: str = "./generation/", output_file: str = "merged_datasets.json"):
    merged_json = []
    for json_file in glob.glob(os.path.join(output_dir, "*.json")):
        with open(json_file, "r") as file:
            merged_json.extend(json.load(file))
    with open(os.path.join(output_dir, output_file), "w") as output_file:
              json.dump(merged_json, output_file)


def load_instructions(file_path: str):
    with open(file_path, "r") as json_file:
        loaded_json = json.load(json_file)
    return loaded_json


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
        'Add a off-by-one error in the loop range.',
        'Remove or misplace indentation in the code.',
        'Use an incorrect operator in the condition.',
        'Omit a return statement in the function.',
        'Replace a commonly used function with a non-existent one.'
    ]
    new_tasks = []
    for task in instructions:
        chosen_method = random.choice(methods)
        prompt = f"Generate a code response for the following instruction with an intentional mistake using the method '{chosen_method}'.\n\nInstruction: {task['instruction']}\n\nResponse with Mistake:"
        response = send_request_to_local_llm(prompt, model, temperature, max_tokens)
        incorrect_response = response["choices"][0]["message"]["content"]
        new_tasks.append({
            "instruction": task['instruction'],
            "output": f"I've written a piece of code based on your instruction, but I've included a small mistake on purpose. Can you figure out what's wrong?\n\n{incorrect_response}"
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
):
    load_dotenv(override=True)
    logging.basicConfig(filename="app.log", filemode="w", format='%(name)s - %(levellevelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    prev_tasks = load_instructions(seed_tasks_path)
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
