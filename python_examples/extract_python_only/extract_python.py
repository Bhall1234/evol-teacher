import json

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_filtered_dataset(filtered_data, output_file_path):
    with open(output_file_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)

def filter_python_examples(dataset):
    filtered_data = []
    for item in dataset:
        instruction = item.get('instruction', '').lower()
        output = item.get('output', '').lower()
        if 'python' in instruction or 'python' in output:
            filtered_data.append(item)
    return filtered_data

def main(input_file_path, output_file_path):
    dataset = load_dataset(input_file_path)
    filtered_data = filter_python_examples(dataset)
    save_filtered_dataset(filtered_data, output_file_path)
    print(f"Filtered dataset saved to {output_file_path}. Total examples: {len(filtered_data)}")

if __name__ == "__main__":
    input_file_path = 'C:/Users/benha/Documents/GitHub/evol-teacher/converted_alpaca_2k.json'
    output_file_path = 'C:/Users/benha/Documents/GitHub/evol-teacher/curated_python_examples.json'
    main(input_file_path, output_file_path)
