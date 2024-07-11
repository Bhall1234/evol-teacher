import os
import json
import glob
import logging
import dataclasses
import random
import time
from typing import Optional, Sequence

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

@dataclasses.dataclass
class OpenAIDecodingArguments:
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
            result.append({"instruction": record["instruction"], "output": record["instances"][0]["output"]})
    else:
        with open(file_path, "r") as json_file:
            loaded_json = json.load(json_file)
        for record in loaded_json:
            if record["input"]:
                record["instruction"] += '\n' + record["input"]
            result.append({"instruction": record["instruction"], "output": record["output"]})
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

def evolve_instructions(instructions, client) -> None:
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
        prompt = f"Please modify the following instruction to include an intentional mistake that the user should identify and correct.\n\nInstruction: {task['instruction']}\n\nModification Method: {chosen_method}\n\nModified Instruction:"

        response = client.chat.completions.create(
            model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
            messages=[{"role": "user", "content": prompt}],
            temperature= 1.0, #original = 0.7
        )
        
        new_instruction = response.choices[0].message["content"]
        if not check_instruction(new_instruction):
            new_tasks.append({"instruction": new_instruction})
    return new_tasks

def generate_responses(instructions, client) -> None:
    new_dataset = []
    for task in instructions:
        response = client.chat.completions.create(
            model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
            messages=[{"role": "user", "content": task["instruction"]}],
            temperature=0.7,
        )
        
        new_output = response.choices[0].message["content"]
        if not check_response(new_output):
            new_dataset.append({
                "instruction": task["instruction"],
                "output": new_output
            })
    return new_dataset

def check_instruction(instruction) -> bool:
    if not instruction:
        return True
    if len(instruction.split()) <= 3:
        return True
    if not instruction[0].isascii():
        return True
    if instruction.count("#Rewritten Prompt#") > 0:
        return True
    return False

def check_response(response) -> bool:
    if not response:
        return True
    if len(response.split()) <= 3:
        return True
    if not response[0].isascii():
        return True
    if "sorry" in response.lower():
        return True
    return False

def generate_evol_instruct_set(
    output_dir="./new_generation/",
    seed_tasks_path="./data/EvolInstruct-Code-80k/converted_alpaca_20k.json",
    evolutions=3,
):
    logging.basicConfig(filename="app.log", filemode="w", format='%(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    prev_tasks = load_instructions(seed_tasks_path)
    start_time = time.time()
    for evolution in range(1, evolutions+1):
        evolution_start_time = time.time()
        print(f'Evolution {evolution}:')

        # 1. Evolving Instructions
        print("Generating New Instructions")
        new_tasks = evolve_instructions(prev_tasks, client)

        # 2. Generating Responses to the New Instructions
        print("Generating New Responses")
        new_dataset = generate_responses(new_tasks, client)

        # 3. Output Evolution to a JSON file
        output_file = output_dir + "evol-instruct-" + str(evolution) + '.json'
        with open(output_file, "w") as json_file:
            json.dump(new_dataset, json_file)
        prev_tasks = new_dataset
        evolution_time = time.time() - evolution_start_time
        print(f'Evolution {evolution} complete, took {evolution_time:.2f}s')
    final_time = time.time() - start_time
    print(f'All Computation complete, total run took {final_time:.2f}s')

if __name__ == "__main__":
    #convert_alpaca_to_evol(file_path="./data/code_alpaca_20k.json", output_file="./data/converted_alpaca_20k.json")
    generate_evol_instruct_set()
    merge_evolutions(output_dir="./generation/")