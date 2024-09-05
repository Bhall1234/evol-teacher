# combine the generated code and the original dataset that has been modified?

import json
import os

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, output_file_path):
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)

def combine_json_files(file_paths, output_file_path):
    combined_data = []
    for file_path in file_paths:
        data = load_json(file_path)
        combined_data.extend(data)
    
    save_json(combined_data, output_file_path)
    print(f'Combined JSON data saved to {output_file_path}')

if __name__ == "__main__":
    # List of JSON files to combine
    json_files = [
        './python_examples/combine_gen_original/diverse_responses_evolution_1.json',
        './python_examples/combine_gen_original/diverse_responses_evolution_2.json',
        './python_examples/combine_gen_original/diverse_responses_evolution_3.json',
        './python_examples/combine_gen_original/generated_python_questions_20240727131756_baoviwnf.json'
    ]
    
    # Output file path for combined data
    output_file_path = './combined_output.json'
    
    combine_json_files(json_files, output_file_path)
