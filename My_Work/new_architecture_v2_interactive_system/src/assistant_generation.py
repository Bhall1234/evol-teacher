import requests

def send_request_to_local_llm(messages, model, temperature=0.8, max_tokens=512): 
    url = "http://localhost:1234/v1/chat/completions"  # This is the URL for the local LLM
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": messages,  # The whole conversation history is passed here
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def generate_explanation(prompt, history, model="TheBloke/CodeLlama-13B-Instruct-GGUF", temperature=0.8, max_tokens=512):
    # Append the new prompt to the history
    history.append({"role": "user", "content": prompt})
    
    # Send the request with the entire conversation history
    response = send_request_to_local_llm(history, model, temperature, max_tokens)
    
    # Extract and append the assistant's response to the history
    assistant_message = response["choices"][0]["message"]["content"].strip()
    history.append({"role": "assistant", "content": assistant_message})
    
    # Return the assistant's message
    return assistant_message, history
