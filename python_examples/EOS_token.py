import json

# Define the EOS token
EOS_TOKEN = " <|endoftext|>"

def add_eos_to_record(record):
    if 'output' in record:
        record['output'] = record['output'].strip() + EOS_TOKEN
    return record

def add_eos_to_dataset(input_file, output_file):
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    updated_dataset = [add_eos_to_record(record) for record in dataset]
    
    with open(output_file, 'w') as f:
        json.dump(updated_dataset, f, indent=4)
    
    print(f"EOS tokens added to dataset and saved to {output_file}")

if __name__ == "__main__":
    input_file_path = './python_examples/answered_questions/outputs/manual_edited_shuffle.json' #python_examples\answered_questions\outputs\paired_python_qa_2024-07-29_23-41-00_csdyeqpg_temp_02_1900_input_eos_v2.json
    output_file_path = './python_examples/answered_questions/outputs\manual_edited_shufflle_EOS.json'
    
    add_eos_to_dataset(input_file_path, output_file_path)