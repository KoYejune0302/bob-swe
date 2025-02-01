import os
import json
from rank_bm25 import BM25Okapi
import re

# Base directories
codebase_dir = "codebase"
input_data_dir = "input_data"
os.makedirs(input_data_dir, exist_ok=True)

# Load the extracted data from the JSON file
with open("swe_bench_lite_dev.json", "r") as f:
    extracted_data = json.load(f)

# Function to extract code snippets from a file
def extract_code_snippets(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Use regex to extract functions or code blocks
    code_snippets = re.findall(r"(def\s+\w+\(.*?\):.*?(?=\n\s*def|\Z))", content, re.DOTALL)
    return code_snippets

# Function to tokenize text
def tokenize(text):
    return re.findall(r"\w+", text.lower())

# Iterate through each entry in the extracted data
for entry in extracted_data:
    instance_id = entry["instance_id"]
    problem_statement = entry["problem_statement"]
    codebase_path = os.path.join(codebase_dir, instance_id)

    # Skip if the codebase directory does not exist
    if not os.path.exists(codebase_path):
        print(f"Codebase directory {codebase_path} does not exist. Skipping instance {instance_id}.")
        continue

    # Collect all code snippets from the codebase with file paths
    all_snippets = []
    for root, _, files in os.walk(codebase_path):
        for file in files:
            if file.endswith(".py"):  # Process only Python files
                file_path = os.path.join(root, file)
                snippets = extract_code_snippets(file_path)
                for snippet in snippets:
                    all_snippets.append((file_path, snippet))

    # Tokenize the problem statement and code snippets
    tokenized_problem = tokenize(problem_statement)
    tokenized_snippets = [tokenize(snippet) for _, snippet in all_snippets]

    # Use BM25 to rank code snippets
    bm25 = BM25Okapi(tokenized_snippets)
    scores = bm25.get_scores(tokenized_problem)

    # Get the top 3 most relevant snippets
    top_n = 3
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    top_snippets = [all_snippets[i] for i in top_indices]

    # Save the extracted input
    input_data_path = os.path.join(input_data_dir, instance_id)
    os.makedirs(input_data_path, exist_ok=True)
    with open(os.path.join(input_data_path, "input.txt"), "w") as f:
        f.write(f"Problem Statement:\n{problem_statement}\n\n")
        f.write("Relevant Code Snippets:\n")
        for i, (file_path, snippet) in enumerate(top_snippets, 1):
            f.write(f"File: {file_path}\n")
            f.write(f"Snippet {i}:\n{snippet}\n\n")

    print(f"Extracted input saved for instance {instance_id} in {input_data_path}.")