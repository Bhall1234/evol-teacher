import json

def add_input_field(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    for record in data:
        # Store the original keys and values
        instruction = record.pop("instruction")
        output = record.pop("output")
        
        # Reinsert the keys in the needed order
        record["instruction"] = instruction
        record["input"] = ""
        record["output"] = output

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

file_path = './python_examples/answered_questions/outputs/manual_edited_shuffle_noEOS_evolved_no_input.json'
add_input_field(file_path)
print("Added 'input' field to each record in the dataset.")
