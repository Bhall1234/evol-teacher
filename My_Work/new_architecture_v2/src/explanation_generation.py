"""
The idea is that I use my local model to generate explanations to user questions on python programming topics.
Then I can combine what the local model says with a follow up piece of incorrect code which will allow them to debug after they have just learnt about that particular topic.

temp may need changing depending on the model in use. May use codellama, or deepseek. Not sure, could try both of these.
The temp and max tokens may need changing depending on how well the model(s) perform.


# Contains functions for generating explanations using the local LLM.

import requests

def send_request_to_local_llm(prompt, model, temperature=0.8, max_tokens=512): 
    url = "http://localhost:1234/v1/chat/completions"  # This is the url for the local LLM
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def generate_explanation(prompt, model, temperature=0.8, max_tokens=512):
    response = send_request_to_local_llm(prompt, model, temperature, max_tokens)
    explanation = response["choices"][0]["message"]["content"].strip()
    explanation = '\n'.join([line.strip() for line in explanation.splitlines() if line.strip()])
    return explanation"""


import requests

def send_request_to_local_llm(prompt, model, temperature=0.8, max_tokens=512): 
    url = "http://localhost:1234/v1/chat/completions"  # This is the url for the local LLM
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def generate_explanation(prompt, model, temperature=0.8, max_tokens=512):
    response = send_request_to_local_llm(prompt, model, temperature, max_tokens)
    explanation = response["choices"][0]["message"]["content"].strip()
    # Preserve the whitespace and indentation for code blocks
    explanation = explanation.replace('\n```', '\n```\n')
    return explanation
