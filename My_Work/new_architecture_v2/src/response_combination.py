# Contains functions for combining the explanation and incorrect code into a single response.

def create_combined_response(user_question, explanation, correct_code):
    combined_response = (f"{explanation}\n\nIncorrect Code:\n```python\n"
                         f"{correct_code}\n```\nPrompt:\nCan you identify the problem in the code?")
    return combined_response
