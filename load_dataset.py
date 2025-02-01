from datasets import load_dataset
import json

# Load the SWE-bench-lite dataset from Hugging Face
dataset = load_dataset("princeton-nlp/SWE-bench_Lite")

# Extract the 'dev' dataset
dev_dataset = dataset['dev']

# Extract the specified columns
columns_to_extract = ["repo", "instance_id", "base_commit", "problem_statement"]
extracted_data = dev_dataset.select_columns(columns_to_extract)

# Convert the extracted data to a list of dictionaries
extracted_data_list = extracted_data.to_list()

# Save the extracted data to a JSON file
output_file = "swe_bench_lite_dev.json"
with open(output_file, 'w') as f:
    json.dump(extracted_data_list, f, indent=4)

print(f"Extracted data saved to {output_file}")