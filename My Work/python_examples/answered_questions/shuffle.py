import json
import random

# Load the JSON data
with open('./python_examples/answered_questions/outputs/paired_python_qa_2024-07-29_23-41-00_csdyeqpg_temp_02_1900_input_eos_v2.json', 'r') as file:
    data = json.load(file)

# Shuffle the data
random.shuffle(data)

# Save the shuffled data back to a JSON file
with open('./python_examples/answered_questions/outputs/paired_python_qa_2024-07-29_23-41-00_csdyeqpg_temp_02_1900_input_eos_shuffle_v2.json', 'w') as file:
    json.dump(data, file, indent=4)
