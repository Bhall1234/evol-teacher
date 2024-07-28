import json

def add_input_field(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    for record in data:
        # Store the original keys and values
        instruction = record.pop("instruction")
        answer = record.pop("answer")
        
        # Reinsert the keys in the desired order
        record["instruction"] = instruction
        record["input"] = ""
        record["answer"] = answer

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
file_path = './python_examples/answered_questions/outputs/paired_python_qa_2024-07-27_15-20-53_ysogztji_1900_alpaca.json'
add_input_field(file_path)
print("Added 'input' field to each record in the dataset.")
