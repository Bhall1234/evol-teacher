import json

def add_input_field(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    for record in data:
        # Store the original keys and values
        instruction = record.pop("instruction")
        output = record.pop("output")
        
        # Reinsert the keys in the desired order
        record["instruction"] = instruction
        record["input"] = ""
        record["output"] = output

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage python_examples\answered_questions\outputs\paired_python_qa_2024-07-29_23-41-00_csdyeqpg_temp_02_1900_v2.json
file_path = './python_examples/answered_questions/outputs/manual_edited_shuffle_noEOS_evolved_no_input.json'
add_input_field(file_path)
print("Added 'input' field to each record in the dataset.")
