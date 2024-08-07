# Contains utility functions, such as loading datasets, saving outputs, and sending requests to the local LLM.

import json

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_dataset(data, output_file_path):
    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=4)
