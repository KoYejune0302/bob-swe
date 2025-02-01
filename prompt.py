# prompt.py

DEFAULT_PROMPT_TEMPLATE = """You are tasked with fixing a python code issue. Below is the problem statement and relevant code snippets with file paths and starting line numbers.

Problem Statement:
{problem_statement}

Relevant Code Snippets:
{formatted_snippets}

Generate a git diff patch that fixes the problem. Follow these guidelines:

1.  **Crucially, use the provided file paths and starting line numbers to create accurate diff patches.** The line numbers indicate the beginning of the code snippet in the original file.
2.  Include the file names, line numbers (using the provided starting line and context from the code snippets), and precise code changes in the diff patch.
3.  Ensure the patch is syntactically correct and follows the project's coding style.
4.  Only output the diff patch. Do not include any other information or explanations.

Now, generate the patch for the given problem:
"""