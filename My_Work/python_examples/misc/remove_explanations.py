import json
import re

# Function to clean the output
def clean_output(output):
    incorrect_code_pattern = re.compile(r"(Incorrect Code:\n```python[\s\S]*?\n```)")
    prompt_pattern = re.compile(r"(Prompt:\nCan you spot[\s\S]*)")
    
    incorrect_code_match = incorrect_code_pattern.search(output)
    prompt_match = prompt_pattern.search(output)
    
    if incorrect_code_match and prompt_match:
        cleaned_output = incorrect_code_match.group(1) + "\n" + prompt_match.group(1)
        return cleaned_output
    return output

def process_data(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    for item in data:
        item['output'] = clean_output(item['output'])
    
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

input_file = 'python_examples/answered_questions/outputs/paired_python_qa_2024-07-29_12-34-06_yrpxmede.json'
output_file = 'python_examples/answered_questions/outputs/removed_explanation_test.json'
process_data(input_file, output_file)
