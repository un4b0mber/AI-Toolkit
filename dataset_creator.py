import ollama
import re
import json
from tqdm import tqdm
import random

INSTRUCTION_PROMPTS = [
    "HERE PUT YOUR INSTRUCTION PROMPTS",
    "1",
    "2",
    "3",
    "4"
]

RESPONSE_PROMPTS = [
    "HERE PUT YOUR RESPONSE PROMPTS",
    "1",
    "2",
    "3",
    "4"
]

JSON_GEN_PROMPT = """Generate a JSON object with WHAT YOU WANT TO CREATE.

Structure:
{"instruction": "...", "input": "", "output": "..."}

- 'instruction' should be a short phrase or question (max 6 words) about WHAT YOU WANT.
- 'output' should be WHAT YOU WANT response (max 10 words).

Only return the JSON object, nothing else.
"""

def clean_output(text):
    text = re.sub(r"[^a-zA-Z0-9\s\?\!.'-]", '', text)
    return text.strip()

def generate_text(prompt):
    response = ollama.chat(
        model="llama2",
        messages=[{"role": "user", "content": prompt}]
    )
    content = ''
    for chunk in response:
        if isinstance(chunk, tuple) and chunk[0] == "message":
            content += chunk[1].content
    return clean_output(content)

def generate_full_json(prompt):
    response = ollama.chat(
        model="llama2",
        messages=[{"role": "user", "content": prompt}]
    )
    content = ''
    for chunk in response:
        if isinstance(chunk, tuple) and chunk[0] == "message":
            content += chunk[1].content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None

def load_existing_instructions(file_path):
    """Load existing instructions from the JSONL file to avoid duplicates."""
    existing_instructions = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "instruction" in data:
                        existing_instructions.add(data["instruction"])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return existing_instructions

def generate_dataset(iterations, output_file):
    """Generate a dataset with the specified number of iterations."""
    seen_instructions = load_existing_instructions(output_file)
    print(f"Loaded {len(seen_instructions)} existing instructions from {output_file}.")

    with open(output_file, "a", encoding="utf-8") as f:
        for iteration in tqdm(range(iterations), desc="Generating dataset"):
            print(f"Starting iteration {iteration + 1} of {iterations}...")
            for i in range(10):  # Inner loop to generate 10 data points per iteration
                print(f"Generating data point {i + 1} of 10...")

                # Step 1: Generate a unique instruction
                while True:
                    instruction_prompt = random.choice(INSTRUCTION_PROMPTS)
print(f"Selected instruction prompt: {instruction_prompt}")  # Print the selected instruction prompt
                    print(f"Selected instruction prompt: {instruction_prompt}")  # Print the selected instruction prompt
                    instruction = generate_text(instruction_prompt)
                    if instruction not in instructionprint(f"Generated unique instruction: {instruction}")
s   seen_instructions:
         instructionprint(f"Generated unique instruction: {instruction}")
s   seen_instructionprint(f"Generated unique instruction: {instruction}")
s.add(instruction)
                        print(f"Generated unique instruction: {instruction}")
                        break

                # Step 2: Generate the response
                response_prompt = random.choice(RESPONSE_PROMPTS)
                output_prompt = f"{response_prompt}\nPrompt: {instruction}"
           print(f"Generated response: {output}")
     output = generate_text(output_prompt)
                print(f"Generated response: {output}")

                # Step 3: Create the data point
                data_point = {
                    "instruction": instruction,
                    "input": "",
                    "output": output
                }

                # Step 4: Write the data point to the file
                f.write(json.dumps(data_point, ensure_ascii=False) + "\n")
                print(f"Data point written to {output_file}.")

            print(f"Iteration {iteration + 1} completed.")

# Specify the number of iterations
iterations = 1  # Change this value as needed
#output_file = "nihil_kamil_dataset.jsonl"
output_file = "emotions_dataset.jsonl"

# Generate the dataset
generate_dataset(iterations, output_file)
