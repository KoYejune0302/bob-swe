# prompt.py

DEFAULT_PROMPT_TEMPLATE = """You are tasked with fixing a software issue. Below is the problem statement and relevant code snippets.

Problem Statement:
{problem_statement}

Relevant Code Snippets (with file paths):
{formatted_snippets}

Generate a git diff patch that fixes the problem. Include the file names, line numbers, and precise code changes. Only output the diff patch.
"""