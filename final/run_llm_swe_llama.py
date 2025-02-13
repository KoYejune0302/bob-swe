# run_llm_swe_llama.py
import os
import json
import re
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from prompt import DEFAULT_PROMPT_TEMPLATE  # Import the updated prompt template

# Configuration
model_name_or_path = "princeton-nlp/SWE-Llama-7b"
input_data_dir = "input_data"
output_dir = "results/swe-llama"
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
output_file = os.path.join(output_dir, f"model_patches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

# Optimize CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# 8-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit quantization
    llm_int8_threshold=6.0,  # Threshold for int8 quantization
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    quantization_config=quantization_config,
    use_cache=False  # Disable caching to save memory
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16,
)

def extract_diff_from_response(response):
    """Extract the diff patch from the model's response."""
    diff_match = re.search(r'--- a/.*', response, re.DOTALL)
    return diff_match.group(0).strip() if diff_match else ""

def remove_codebase_prefix(patch, instance_id):
    """Remove the 'codebase/{instance_id}/' prefix from the patch."""
    return patch.replace(f"codebase/{instance_id}/", "")

def process_instance(instance_id):
    """Process a single instance to generate a patch."""
    input_path = os.path.join(input_data_dir, instance_id, "input.txt")
    if not os.path.exists(input_path):
        return None

    with open(input_path, "r") as f:
        content = f.read()

    # Parse problem statement and snippets
    problem_statement = []
    snippets = []
    current_section = None
    current_file = None
    current_snippet_data = None

    for line in content.split("\n"):
        if line.startswith("Problem Statement:"):
            current_section = "problem"
            problem_statement.append(line[len("Problem Statement:"):].strip())
        elif line.startswith("File: "):
            current_file = line[len("File: "):].strip()
            current_section = "snippet_file"
        elif line.startswith("Snippet "):
            match = re.match(r"Snippet \d+, Line Start: (\d+):", line)
            if match:
                start_line = int(match.group(1))
                current_snippet_data = {"file": current_file, "start_line": start_line, "code": []}
                snippets.append(current_snippet_data)
                current_section = "snippet_code"
        elif current_section == "problem":
            problem_statement.append(line.strip())
        elif current_section == "snippet_code" and snippets and current_snippet_data is not None:
            if line.strip() != "": # Ignore empty lines within code snippet for prompt formatting
                current_snippet_data["code"].append(line)


    # Format sections
    problem_statement = "\n".join(problem_statement)
    formatted_snippets = []
    for snippet in snippets:
        formatted_snippets.append(
            f"File: {snippet['file']}, Start Line: {snippet['start_line']}\n" +
            "```python\n" +
            "\n".join(snippet['code']) +
            "\n```"
        )
    formatted_snippets = "\n\n".join(formatted_snippets)


    # Generate prompt
    prompt = DEFAULT_PROMPT_TEMPLATE.format(
        problem_statement=problem_statement,
        formatted_snippets=formatted_snippets
    )


    # Generate response
    response = pipe(
        prompt,
        max_new_tokens=4096,  # Increased max_new_tokens to 4096
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,
        batch_size=4  # Reduce batch size to save memory
    )[0]['generated_text']


    # Extract diff
    model_patch = extract_diff_from_response(response)

    # Ensure model_patch is always a string, even if extraction fails
    if not model_patch:
        model_patch = "No patch generated" # Or you can use an empty diff: "No patch generated" # Or you can use an empty diff: ""


    # Remove the 'codebase/{instance_id}/' prefix
    model_patch = remove_codebase_prefix(model_patch, instance_id)

    return {
        "instance_id": instance_id,
        "model_patch": model_patch,
        "model_name_or_path": "YejuneKo/SWE-Llama-7b",
    }

# Process all instances incrementally
results = []
if os.path.exists(output_file):
    # Load existing results if the file already exists
    with open(output_file, "r") as f:
        results = json.load(f)

# Get list of instance IDs
instance_ids = [instance_id for instance_id in os.listdir(input_data_dir) if os.path.isdir(os.path.join(input_data_dir, instance_id))]

# Process each instance and save results incrementally
for instance_id in instance_ids:
    try:
        # Skip if already processed
        if any(result["instance_id"] == instance_id for result in results):
            print(f"Skipping already processed instance: {instance_id}")
            continue

        # Process the instance
        result = process_instance(instance_id)
        if result:
            results.append(result)
            print(f"Processed {instance_id}")

        # Save results after each iteration
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            print(f"Results saved incrementally to {output_file}")

    except Exception as e:
        print(f"Error processing instance {instance_id}: {e}")
        # Save results even if an error occurs
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            print(f"Partial results saved to {output_file}")

print(f"Final results saved to {output_file}")