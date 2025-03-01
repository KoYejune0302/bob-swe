import os
import json
from git import Repo

# Path to the JSON file generated by load_dataset.py
input_file = "swe_bench_verified_test.json"

# Base directory to save codebases
base_dir = "codebase"
os.makedirs(base_dir, exist_ok=True)

# Load the extracted data from the JSON file
with open(input_file, 'r') as f:
    extracted_data = json.load(f)

# Iterate through each entry in the extracted data
for entry in extracted_data:
    repo_url = f"https://github.com/{entry['repo']}.git"  # Construct the GitHub URL
    instance_id = entry['instance_id']
    base_commit = entry['base_commit']
    codebase_dir = os.path.join(base_dir, instance_id)

    # Skip if the directory already exists
    if os.path.exists(codebase_dir):
        print(f"Directory {codebase_dir} already exists. Skipping clone for instance {instance_id}.")
        continue

    # Clone the repository at the specified base commit
    print(f"Cloning {repo_url} at commit {base_commit} into {codebase_dir}...")
    try:
        # Clone the repository
        repo = Repo.clone_from(repo_url, codebase_dir)
        # Check out the specific base commit
        repo.git.checkout(base_commit)
        print(f"Successfully cloned and checked out commit {base_commit} for instance {instance_id}.")
    except Exception as e:
        print(f"Failed to clone or checkout {repo_url} at commit {base_commit}: {e}")