import json

def remove_input_field(input_file_path, output_file_path):
    # Load the dataset
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    # Remove the 'input' field from each item
    for item in data:
        if 'input' in item:
            del item['input']

    # Save the modified dataset to a new file
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    input_file_path = './python_examples/answered_questions/outputs/manual_edited_shuffle_noEOS_smaller.json'
    output_file_path = './python_examples/answered_questions/outputs/manual_edited_shuffle_noEOS_smaller_no_input.json'
    
    remove_input_field(input_file_path, output_file_path)
    print(f"Modified dataset saved to {output_file_path}")
