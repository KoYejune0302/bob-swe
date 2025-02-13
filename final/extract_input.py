import os
import json
from rank_bm25 import BM25Okapi
import re

# Base directories
codebase_dir = "codebase"
input_data_dir = "input_data"
os.makedirs(input_data_dir, exist_ok=True)

# Load the extracted data from the JSON file
with open("swe_bench_verified_test.json", "r") as f:
    extracted_data = json.load(f)

# Function to extract code snippets with line numbers and indentation from a file
def extract_code_snippets_with_context(file_path):
    snippets = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        in_code_block = False
        current_snippet = []
        start_line_number = -1

        for line_number, line in enumerate(lines, 1):
            stripped_line = line.strip()
            if stripped_line.startswith("def ") or stripped_line.startswith("class "):  # Heuristics to detect code block starts
                if current_snippet:  # Save previous snippet if exists
                    snippets.append({
                        "start_line": start_line_number,
                        "code": current_snippet
                    })
                    current_snippet = []
                in_code_block = True
                current_snippet.append(line.rstrip())  # Keep original indentation and line endings
                start_line_number = line_number
            elif in_code_block:
                # Check if next line exists before accessing it
                if stripped_line == "" and len(current_snippet) > 0 and line_number < len(lines) and lines[line_number].strip() == "":
                    snippets.append({
                        "start_line": start_line_number,
                        "code": current_snippet
                    })
                    current_snippet = []
                    in_code_block = False
                    start_line_number = -1
                else:
                    current_snippet.append(line.rstrip())  # Keep original indentation and line endings
        if current_snippet:  # For the last code block if file ends within a block
            snippets.append({
                "start_line": start_line_number,
                "code": current_snippet
            })
    return snippets

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

    # Collect all code snippets from the codebase with file paths and line numbers
    all_snippets = []
    for root, _, files in os.walk(codebase_path):
        # Skip directories named 'test' or 'tests'
        if "test" in root.lower() or "tests" in root.lower():
            continue

        for file in files:
            if file.endswith(".py"):  # Process only Python files
                file_path = os.path.join(root, file)
                snippets_with_context = extract_code_snippets_with_context(file_path)
                for snippet_data in snippets_with_context:
                    all_snippets.append((file_path, snippet_data))

    # Tokenize the problem statement and code snippets (using only code content for BM25)
    tokenized_problem = tokenize(problem_statement)
    tokenized_snippets_for_bm25 = [tokenize("\n".join(snippet_data['code'])) for _, snippet_data in all_snippets]

    # Check if any snippets were extracted to avoid division by zero error
    if not tokenized_snippets_for_bm25:
        print(f"No code snippets found for instance {instance_id}. Skipping BM25 ranking.")
        continue

    # Use BM25 to rank code snippets
    bm25 = BM25Okapi(tokenized_snippets_for_bm25)
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
        for i, (file_path, snippet_data) in enumerate(top_snippets, 1):
            f.write(f"File: {file_path}\n")
            f.write(f"Snippet {i}, Line Start: {snippet_data['start_line']}:\n")
            for code_line in snippet_data['code']:
                f.write(f"{code_line}\n")
            f.write("\n")

    print(f"Extracted input saved for instance {instance_id} in {input_data_path}.")
