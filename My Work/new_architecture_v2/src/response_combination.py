# Contains functions for combining the explanation and incorrect code into a single response.

from src.incorrect_code_generation import introduce_errors

def create_combined_response(user_question, explanation, correct_code):
    incorrect_code = introduce_errors(correct_code)
    combined_response = (f"{explanation}\n\nIncorrect Code:\n```python\n"
                         f"{incorrect_code}\n```\nPrompt:\nCan you identify the problem in the code?")
    return combined_response
