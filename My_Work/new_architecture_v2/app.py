import logging
import random
import re
import os
import tempfile
import subprocess
import sys
import uuid
from flask import Flask, request, render_template, jsonify, session
from src.utils import load_dataset
from src.explanation_generation import generate_explanation
#from src.assistant_generation import generate_explanation
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import spacy
from fuzzywuzzy import fuzz
from spacy.matcher import PhraseMatcher
from openai import OpenAI

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session management
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

@app.before_request
def assign_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())  # Generate a unique session ID

def get_session_id():
    return session.get('session_id', 'UnknownSession')

def log_with_session(message, level=logging.INFO):
    session_id = get_session_id()
    task_id = session.get('current_task_id', 'UnknownTask')
    logging.log(level, f"Session ID: {session_id}, Task ID: {task_id} - {message}")

# Set up logging
log_path = os.path.join(os.getcwd(), 'My_Work', 'new_architecture_v2', 'src', 'logs', 'user_interactions.log')
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load datasets
correct_code_examples = load_dataset('./My_Work/new_architecture_v2/data/code_examples.json')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize PhraseMatcher
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# Add patterns to the matcher
phrases = ["for loop", "while loop", "if statement", "if statements","hash table", 
           "key-value", "conditional statement", "key value pair", "conditional statements",
           "error handling", "try except", "exception handling", "simple exercises", "simple problem", 
           "simple problems", "simple task", "simple tasks", "comparison operators", "logical operators",
           "data types", "types of data"]
patterns = [nlp.make_doc(phrase) for phrase in phrases]
phrase_matcher.add("PROGRAMMING_PHRASES", patterns)

# Synonyms for better keyword matching
synonyms = {
    "append": ["add", "insert", "push"],
    "list": ["array", "vector"],
    "dictionary": ["dict", "map", "hash map", "hash table", "key-value"],
    "tuple": ["pair", "ordered pair"],
    "open": ["read"],
    "write": ["save"],
    "import": ["include"],
    "while loop": ["while_loop", "`while`", "while"],
    "for loop": ["for_loop", "for"],
    "hash table": ["hash_map", "hashmap", "hash"],
    "key-value": ["key value", "key-value pair", "key-value pairs"], 
}

@app.route("/")
def home():
    log_with_session("Accessed the home page.")
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    predefined_question = request.form.get("predefined_question")
    custom_question = request.form.get("question")

    # Use the predefined question if selected; otherwise, use the custom question
    user_question = predefined_question if predefined_question else custom_question
    log_with_session(f"User question received: {user_question}")

    # Reset the context for the chat conversation when a new question is asked.
    session.clear()

    if "python" not in user_question.lower():
        user_question += " (programming Language: python)"
        log_with_session(f"Modified user question: {user_question}")
    
    explanation = generate_explanation(user_question, "TheBloke/CodeLlama-13B-Instruct-GGUF")
    log_with_session(f"Generated explanation: {explanation}")
    
    initial_explanation = extract_initial_explanation(explanation)
    log_with_session(f"Initial part of the explanation: {initial_explanation}")
    
    user_keywords = extract_programming_keywords(user_question)
    explanation_keywords = extract_programming_keywords(initial_explanation)
    combined_keywords = user_keywords.union(explanation_keywords)
    log_with_session(f"Extracted user keywords: {user_keywords}")
    log_with_session(f"Extracted explanation keywords: {explanation_keywords}")
    log_with_session(f"Combined keywords: {combined_keywords}")
    
    incorrect_code_data = get_related_code_by_keywords(combined_keywords, correct_code_examples)
    log_with_session(f"Selected incorrect code example: {incorrect_code_data}")
    
    session['current_task_id'] = incorrect_code_data.get('task_id', 'No task ID found')
    log_with_session(f"Task ID set to: {session['current_task_id']}")
    
    formatted_explanation = format_code_snippets(explanation)
    formatted_incorrect_code = highlight(incorrect_code_data.get("incorrect_code", "No incorrect code found."), PythonLexer(), HtmlFormatter(noclasses=True))
    formatted_correct_code = highlight(incorrect_code_data.get("correct_code", "No correct code found."), PythonLexer(), HtmlFormatter(noclasses=True))
    
    return render_template("index.html", question=user_question, explanation=formatted_explanation, 
                           incorrect_code=formatted_incorrect_code, correct_code=formatted_correct_code,
                           task_description=incorrect_code_data.get("task_description", "No task description found."),
                           hint=incorrect_code_data.get("description", "No hint found."), 
                           detailed_explanation=incorrect_code_data.get("explanation", "No detailed explanation found."),
                           task_id=incorrect_code_data.get("task_id", "No task ID found."))

@app.route("/interact", methods=["POST"])
def interact():
    try:
        # Log the received data
        data = request.get_json()
        log_with_session(f"Received data: {data}")
        
        user_message = data.get("message")
        task_id = data.get("task_id")
        log_with_session(f"User message: {user_message}, Task ID: {task_id}")
        
        # Initialize or retrieve the conversation history from the session
        history = session.get(f"conversation_history_{task_id}", [])
        log_with_session(f"Current conversation history: {history}")
        
        # Add the user's new message to the conversation history if it's not empty
        if user_message:
            history.append({"role": "user", "content": user_message})
            log_with_session(f"Updated conversation history after user message: {history}")
        
        # If it's the initial code submission
        if "code" in data:
            user_code = data["code"]
            session['current_task_id'] = task_id
            session['user_code'] = user_code  # Store the user's code in session
            log_with_session(f"User code submitted: {user_code}")
            
            correct_example = find_correct_example(task_id, correct_code_examples)
            if not correct_example:
                log_with_session("No matching task found.", level=logging.ERROR)
                return jsonify({"result": "No matching task found."}), 404

            expected_output = correct_example["expected_output"]
            reflection_context = correct_example.get("reflection_context", "")
            log_with_session(f"Expected output: {expected_output}, Reflection context: {reflection_context}")

            try:
                # Run the user code and capture output
                result = subprocess.run([sys.executable, "-c", user_code], capture_output=True, text=True, check=True)
                user_output = result.stdout.strip()  # Ensure any extra whitespace is removed
                log_with_session(f"User output: {user_output}")
            except subprocess.CalledProcessError as e:
                user_output = e.stderr.strip()
                log_with_session(f"Error in user code execution: {user_output}", level=logging.ERROR)
                return jsonify({"result": "Incorrect", "error": user_output}), 200

            # Compare the actual output with the expected output
            if user_output == expected_output.strip():
                result = "Correct"
                log_with_session("User code is correct.")
            else:
                result = "Incorrect"
                log_with_session(f"Expected output: {expected_output}, but got: {user_output}")
            
            response = {"result": result}
            log_with_session(f"Initial response: {response}")

            if result == "Correct":
                # Generate a reflection question based on the user's code
                history.append({
                    "role": "system", 
                    "content": f"Reflection Context to help you create the question:\n{reflection_context}"
                })
                history.append({
                    "role": "system",
                    "content": f"User's Submitted Code:\n{user_code}"
                })
                log_with_session(f"Updated conversation history with reflection context and code: {history}")

                # Generate the completion using the correct object attribute access
                completion = client.chat.completions.create(
                    model="TheBloke/CodeLlama-13B-Instruct-GGUF",
                    messages=history,
                    temperature=0.8,
                    stream=False
                )
                log_with_session(f"Completion response: {completion}")

                # Add assistant's response to history
                reflection_question = completion.choices[0].message.content.strip()
                log_with_session(f"Generated reflection question: {reflection_question}")
                history.append({"role": "assistant", "content": reflection_question})

                # Store the updated conversation history
                session[f"conversation_history_{task_id}"] = history
                log_with_session(f"Stored updated conversation history: {history}")

                response["show_reflection_chat"] = True
                response["initial_chat_message"] = reflection_question
                log_with_session(f"Final response with reflection chat: {response}")

            return jsonify(response)
        
        # Continuation of the conversation
        else:
            log_with_session("Continuing the conversation.")
            completion = client.chat.completions.create(
                model="TheBloke/CodeLlama-13B-Instruct-GGUF",
                messages=history,
                temperature=0.7,
                stream=False
            )
            log_with_session(f"Completion response: {completion}")

            # Fixing the access to the content field
            assistant_response = completion.choices[0].message.content.strip()
            log_with_session(f"Assistant's response: {assistant_response}")
            history.append({"role": "assistant", "content": assistant_response})
            
            # Store the updated conversation history
            session[f"conversation_history_{task_id}"] = history
            log_with_session(f"Stored updated conversation history: {history}")

            return jsonify({"response": assistant_response})

    except Exception as e:
        log_with_session(f"Exception occurred: {str(e)}", level=logging.ERROR)
        import traceback
        traceback.print_exc()  # Print the error to the logs
        return jsonify({"result": "An error occurred", "error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message")
    task_id = request.form.get("task_id")
    log_with_session(f"User chat message: {user_message} for task ID: {task_id}")

    # Retrieve the conversation state
    context_provided = session.get(f"context_provided_{task_id}", False)

    # Find the correct example based on the task ID
    correct_example = find_correct_example(task_id, correct_code_examples)

    if correct_example:
        if not context_provided:
            chat_context = correct_example.get("chat_context", "")
            # Create the initial prompt with context
            prompt = f"Context: {chat_context}\nUser's Question: '{user_message}'\nPlease help the user understand the task without giving out the solution to the problem."
            log_with_session(f"Initial chat prompt with context: {prompt}")
            # Mark that context has been provided
            session[f"context_provided_{task_id}"] = True
        else:
            # Create the prompt without context
            prompt = f"User's Question: '{user_message}'\nPlease provide an explanation."

        # Generate explanation using the refined prompt
        explanation = generate_explanation(prompt, "TheBloke/CodeLlama-13B-Instruct-GGUF")
        log_with_session(f"Generated chat explanation: {explanation}")

        # Format the explanation for display
        formatted_explanation = format_code_snippets(explanation)
        log_with_session(f"Formatted chat explanation: {formatted_explanation}")

        return jsonify({"response": formatted_explanation})
    else:
        log_with_session(f"No matching task found for task ID: {task_id}", level=logging.ERROR)
        return jsonify({"response": "Sorry, I couldn't find any information about this task."})

def run_static_analysis(user_code):
    log_with_session("Entered run_static_analysis function.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(user_code.encode('utf-8'))
            temp_file_name = temp_file.name

        result = subprocess.run(
            ['pylint', '--disable=C0114,C0116,C0304', temp_file_name],
            capture_output=True,
            text=True,
            check=False
        )
        filtered_output = parse_pylint_output(result.stdout)
        log_with_session(f"Completed static analysis. Output:\n{filtered_output}")
        log_with_session(f"Static analysis error:\n{result.stderr}", level=logging.ERROR)

        return filtered_output + result.stderr
    except subprocess.CalledProcessError as e:
        log_with_session(f"Static analysis failed with error:\n{e.stderr}", level=logging.ERROR)
        return e.stderr
    finally:
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

@app.route("/run_code", methods=["POST"])
def run_code():
    try:
        data = request.get_json()
        user_code = data.get("code", "")

        if not user_code:
            return jsonify({"output": "No code provided."}), 400

        # Run the user code using subprocess
        result = subprocess.run([sys.executable, "-c", user_code], capture_output=True, text=True, check=True, timeout=5)
        user_output = result.stdout.strip()

        return jsonify({"output": user_output}), 200

    except subprocess.CalledProcessError as e:
        return jsonify({"output": f"Error in code execution:\n{e.stderr.strip()}"}), 400
    except Exception as e:
        return jsonify({"output": f"Unexpected error:\n{str(e)}"}), 500

def parse_pylint_output(output):
    lines = output.splitlines()
    filtered_lines = []
    for line in lines:
        if line.startswith("*************"):
            continue
        match = re.match(r'.*:\d+:\d+: (.+)', line)
        if match:
            filtered_lines.append(match.group(1))
        else:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)

def extract_initial_explanation(explanation):
    sentences = re.split(r'[.:]\s*', explanation)
    return '. '.join(sentences[:2])

def extract_programming_keywords(text):
    doc = nlp(text)
    keywords = set()

    # Match multi-word phrases like "for loop", "while loop"
    matches = phrase_matcher(doc)
    matched_phrases = [doc[start:end].text.replace(" ", "_") for match_id, start, end in matches]
    keywords.update(matched_phrases)

    # Existing keywords and synonyms logic
    programming_keywords = {
        "def", "function", "return", "functions",
        "else", "elif", "class", "import", "try", "except",
        "concatenate", "append", "add", "subtract", "multiply", "divide",
        "variable", "list", "dictionary", "set",
        "tuple", "range", "open", "read", "write", "combine",
        "hash map", "key error", "hash", "hash table", "modulus",
        "arithmetic", "maths", "remainder", "modulo", "integer",
        "float", "string", "integer division", "while loop", "for loop", 
        "key-value", "key value", "key value pair", "key-value pair",
        "if_statement", "if statement", "for_loop", "while_loop", "hash_table", "key-value",
        "addition", "subtraction", "multiplication", "division","+","*","/","%","//",
        "conditions", "conditional statement", "conditional statements", "`while`", "while_loop",
        "if_statement", "if statements", "for_loop", "for loop", "introductory", "simple exercises", 
        "simple problem", "simple problems", "simple task", "simple tasks", "fundamentals", "basic", "starter", "introductory",
        "exception handling", "error handling", "try except", "exception", "error", "handling", "except",
        "boolean", "bool", "comparison", "comparison operators", "logical", "logical operators", "basics", "introduction",
        "variables", "variable", "data types", "data type", "data","data_type", "types_of_data", "data_types"
    }

    for token in doc:
        lemma = token.lemma_
        if lemma in synonyms:
            keywords.update(synonyms[lemma])
        if lemma in programming_keywords:
            keywords.add(lemma)
    
    log_with_session(f"Extracted programming keywords: {keywords}")
    return keywords

def get_related_code_by_keywords(keywords, correct_code_examples):
    log_with_session(f"Matching keywords: {keywords}")
    
    matching_examples = []
    best_match_score = 0
    best_match_example = None

    for category, data in correct_code_examples.items():
        logging.debug(f"Processing category: {category} with data: {data}")
        
        if isinstance(data, dict) and "label" in data and "examples" in data:
            for key in data["label"]:
                fuzzy_scores = [(fuzz.partial_ratio(keyword, key), keyword) for keyword in keywords]
                
                for score, keyword in fuzzy_scores:
                    logging.debug(f"Fuzzy matching score between '{keyword}' and '{key}': {score}")
                    
                    # Only update if the new score is higher than the current best match
                    if score > best_match_score:
                        best_match_score = score
                        best_match_example = random.choice(data["examples"])

                    # Optionally, add a threshold for matching to avoid irrelevant selections
                    if score > 80:
                        matching_examples.extend(data["examples"])
        
    if not best_match_example:
        log_with_session("No related code examples found.", level=logging.WARNING)
        return {"incorrect_code": "No related code examples found.", "task_description": "", "description": "", "explanation": "", "task_id": "N/A"}
    
    log_with_session(f"Selected example based on highest match score: {best_match_example}")
    return best_match_example

def find_correct_example(task_id, correct_code_examples):
    log_with_session(f"Finding correct example for task ID: {task_id}")
    for category, data in correct_code_examples.items():
        logging.debug(f"Processing category: {category} with data: {data}")

        if isinstance(data, dict) and "examples" in data:
            examples = data["examples"]

            if isinstance(examples, list):
                correct_example = next((ex for ex in examples if str(ex["task_id"]) == str(task_id)), None)
                
                if correct_example:
                    log_with_session(f"Found correct example for task ID {task_id}: {correct_example}")
                    return correct_example
            else:
                logging.warning(f"'examples' in category '{category}' is not a list: {examples}")
        else:
            logging.warning(f"Data in category '{category}' is not a dictionary or lacks 'examples' key: {data}")
    
    log_with_session(f"No correct example found for task ID: {task_id}", level=logging.ERROR)
    return None

def format_code_snippets(response):
    code_block_pattern = re.compile(r'```(?:python)?\n(.*?)\n```', re.DOTALL)
    def replace_code_block(match):
        code = match.group(1).strip()
        highlighted_code = highlight(code, PythonLexer(), HtmlFormatter(noclasses=True))
        return f"<pre><code>{highlighted_code}</code></pre>"
    return code_block_pattern.sub(replace_code_block, response)

if __name__ == "__main__":
    app.run(debug=True)

















# ORIGINAL
"""@app.route("/check_code", methods=["POST"])
def reflection_chat():
    user_message = request.form.get("message")
    task_id = request.form.get("task_id")
    log_with_session(f"User reflection message: {user_message} for task ID: {task_id}")

    # Retrieve the user's correct code and initial reflection question from the session
    user_code = session.get(f"user_code_{task_id}", "")
    initial_reflection_question = session.get(f"initial_reflection_question_{task_id}", "")
    log_with_session(f"User's submitted code: {user_code}")
    log_with_session(f"Initial reflection question: {initial_reflection_question}")

    # Ensure we're generating a follow-up, not a new initial reflection question
    if initial_reflection_question:
        prompt = (
            f"User's Message: '{user_message}'\n"
            f"Initial Reflection Question:\n{initial_reflection_question}\n"
            f"Given the user's message and the context of the initial reflection question, provide a follow-up that either helps clarify the user's understanding or probes deeper into their comprehension of the concept. "
            f"Ensure the follow-up ties directly to the user's response and the initial reflection question."
        )
    else:
        # Fallback if the context is missing (shouldn't happen under normal circumstances)
        prompt = f"User's Message: '{user_message}'\nPlease continue the conversation to support the user's reflection."

    explanation = generate_explanation(prompt, "TheBloke/CodeLlama-13B-Instruct-GGUF")
    log_with_session(f"Generated reflection explanation: {explanation}")

    if explanation:
        # Format and return the response
        formatted_explanation = format_code_snippets(explanation)
        log_with_session(f"Formatted reflection explanation: {formatted_explanation}")
        return jsonify({"response": formatted_explanation})
    else:
        log_with_session("Failed to generate reflection explanation.", level=logging.ERROR)
        return jsonify({"response": "Sorry, I couldn't generate a reflection response."})"""

# NOT ORIGINAL
"""@app.route("/check_code", methods=["POST"])
def check_code():
    try:
        data = request.get_json()
        user_code = data["code"]
        task_id = data["task_id"]
        session['current_task_id'] = task_id
        session['user_code'] = user_code  # Store user's code in session
        log_with_session(f"Received task ID: {task_id}")
        
        correct_example = find_correct_example(task_id, correct_code_examples)
        if not correct_example:
            log_with_session("No matching task found.", level=logging.ERROR)
            return jsonify({"result": "No matching task found."}), 404
        
        expected_output = correct_example["expected_output"]
        log_with_session(f"Expected output: {expected_output}")
        log_with_session(f"User code:\n{user_code}")
        
        try:
            result = subprocess.run([sys.executable, "-c", user_code], capture_output=True, text=True, check=True)
            user_output = result.stdout
            log_with_session(f"User code output: {user_output}")
        except subprocess.CalledProcessError as e:
            user_output = e.stderr
            log_with_session(f"User code execution error: {user_output}", level=logging.ERROR)
        
        result = "Correct" if user_output.strip() == expected_output.strip() else "Incorrect"
        log_with_session(f"Code check result: {result}")
        static_analysis_result = run_static_analysis(user_code)
        
        response = {
            "result": result,
            "static_analysis": static_analysis_result,
        }
        
        if result == "Correct":
            # Store reflection context in session
            session[f"user_code_{task_id}"] = user_code
            session[f"context_provided_{task_id}_reflection"] = False

            # Generate the reflection question immediately
            reflection_context = correct_example.get("reflection_context", "")
            prompt = (
                f"Context: {reflection_context}\n" 
                f"User's Submitted Code:\n{user_code}\n"
                f"Given the user's submitted code, ask a reflection question that probes the user's understanding of the code they submitted in the context given. What is a good question to ask them to deepen their understanding of this code? CONCISE."
                f"Please focus on asking a concise, targeted question related to probing the user's understanding of the code. Do not pad the question with unnecessary information such as 'this is a reflection question or 'Reflection Question:'."
                f"YOU ARE ASKING A QUESTION NOT EXPLAINING THE CONCEPT BEHING THE CODE OR THE CONTEXT. USE THESE TO HELP YOU FORMULATE A QUESTION."
                f"DO NOT REPEAT THE USERS CODE OR THE CONTEXT IN THE QUESTION!!!"
            )

            initial_question = generate_explanation(prompt, "TheBloke/CodeLlama-13B-Instruct-GGUF")
            log_with_session(f"Generated initial reflection question: {initial_question}")

            # Store initial reflection question in session
            session[f"initial_reflection_question_{task_id}"] = initial_question.strip()

            response["show_reflection_chat"] = True
            response["initial_chat_message"] = initial_question.strip()

        return jsonify(response)
    except Exception as e:
        log_with_session(f"Error in check_code: {e}", level=logging.ERROR)
        return jsonify({"result": "An error occurred", "error": str(e)}), 500"""
    
# ORIGINAL
"""@app.route("/reflection_chat", methods=["POST"])
def reflection_chat():
    user_message = request.form.get("message")
    task_id = request.form.get("task_id")
    log_with_session(f"User reflection message: {user_message} for task ID: {task_id}")

    # Retrieve the user's correct code and initial reflection question from the session
    user_code = session.get(f"user_code_{task_id}", "")
    initial_reflection_question = session.get(f"initial_reflection_question_{task_id}", "")
    log_with_session(f"User's submitted code: {user_code}")
    log_with_session(f"Initial reflection question: {initial_reflection_question}")

    # Handle the follow-up reflections based on previous context
    if initial_reflection_question:
        prompt = (
            f"User's Message: '{user_message}'\n"
            #f"Given the context of the user's code submission and the initial reflection question, continue the conversation by asking a follow-up question that challenges the user's understanding. "
            #f"Focus on probing deeper into the concepts or potential edge cases that the user might not have considered. Avoid simply rephrasing the user's response."
            #f"User's Submitted Code:\n{user_code}\n"
            f"Initial Reflection Question:\n{initial_reflection_question}\n"
            #f"Please provide a follow-up reflection question or statement that delves into a related but deeper aspect of the topic."
            f"Follow-up with a statement that delves into a related but deeper aspect of the topic. Help the user if they are stuck or dont seem to understand the concept."
        )
    else:
        # Fallback if the context is missing (shouldn't happen under normal circumstances)
        prompt = f"User's Message: '{user_message}'\nPlease continue the conversation to support the user's reflection."

    explanation = generate_explanation(prompt, "TheBloke/CodeLlama-13B-Instruct-GGUF")
    log_with_session(f"Generated reflection explanation: {explanation}")

    if explanation:
        # Format and return the response
        formatted_explanation = format_code_snippets(explanation)
        log_with_session(f"Formatted reflection explanation: {formatted_explanation}")
        return jsonify({"response": formatted_explanation})
    else:
        log_with_session("Failed to generate reflection explanation.", level=logging.ERROR)
        return jsonify({"response": "Sorry, I couldn't generate a reflection response."})"""

# GARBAGE DOESNT WORK NOT ORIGINAL
"""@app.route("/reflection_chat", methods=["POST"])
def reflection_chat():
    user_message = request.form.get("message")
    task_id = request.form.get("task_id")
    log_with_session(f"User reflection message: {user_message} for task ID: {task_id}")

    # Retrieve the user's correct code and initial reflection question from the session
    user_code = session.get(f"user_code_{task_id}", "")
    initial_reflection_question = session.get(f"initial_reflection_question_{task_id}", "")
    log_with_session(f"User's submitted code: {user_code}")
    log_with_session(f"Initial reflection question: {initial_reflection_question}")

    # Ensure we're generating a follow-up, not a new initial reflection question
    if initial_reflection_question:
        #prompt = (
            #f"User's Message: '{user_message}'\n"
            #f"Initial Reflection Question:\n{initial_reflection_question}\n"
            #f"Provide a follow-up question that deepens the user's understanding of the concept. "
            #f"Do not repeat the user's response or include unnecessary formalities. Focus on the technical aspects."
        #)
        #prompt = (
            #f"User's Message: '{user_message}'\n"
            #f"Initial Reflection Question:\n{initial_reflection_question}\n"
            #f"Given the user's understanding, provide a follow-up question that builds on their response. Avoid simply restating the concept of case sensitivity. Instead, focus on introducing new aspects of the topic, such as potential pitfalls, best practices, or real-world applications.")
        prompt = f"User's Message: '{user_message}'\nPlease continue the conversation to support the user's reflection."
        #prompt = (
            #f"User's Message: '{user_message}'CONTEXT FOR YOU\n"
            #f"Initial Reflection Question:\n{initial_reflection_question} CONTEXT FOR YOU. THIS WAS GENERATED EARILER, YOU ARE TO CONTINUE THE CONVERSATION WITH THE USER.\n"
            #f"Provide a follow-up question that challenges the user to apply their understanding to real-world scenarios, potential pitfalls, or related concepts. "
            #f"Avoid repeating the same theme, and focus on pushing the user's thinking further."
            #f"Do not repeat the user's response or the initial reflection question."
        #)
    else:
        # Fallback if the context is missing (shouldn't happen under normal circumstances)
        prompt = f"User's Message: '{user_message}'\nPlease continue the conversation to support the user's reflection."

    explanation = generate_explanation(prompt, "TheBloke/CodeLlama-13B-Instruct-GGUF")
    log_with_session(f"Generated reflection explanation: {explanation}")

    if explanation:
        # Format and return the response
        formatted_explanation = format_code_snippets(explanation)
        log_with_session(f"Formatted reflection explanation: {formatted_explanation}")
        return jsonify({"response": formatted_explanation})
    else:
        log_with_session("Failed to generate reflection explanation.", level=logging.ERROR)
        return jsonify({"response": "Sorry, I couldn't generate a reflection response."})"""