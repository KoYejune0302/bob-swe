import os
import json
import re
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from prompt import DEFAULT_PROMPT_TEMPLATE  # Import the prompt template

# Configuration
model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Use DeepSeek-R1
input_data_dir = "input_data"
output_file = f"results/deepseek-r1/model_patches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"  # Add timestamp

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    torch_dtype=torch.float16,  # Use 16-bit precision for better performance
    low_cpu_mem_usage=True  # Optimize CPU memory usage
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

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
    
    for line in content.split("\n"):
        if line.startswith("Problem Statement:"):
            current_section = "problem"
            problem_statement.append(line[len("Problem Statement:"):].strip())
        elif line.startswith("File: "):
            current_file = line[len("File: "):].strip()
            current_section = "snippet_file"
        elif line.startswith("Snippet "):
            current_section = "snippet_code"
            snippets.append({"file": current_file, "code": []})
        elif current_section == "problem":
            problem_statement.append(line.strip())
        elif current_section == "snippet_code" and snippets:
            snippets[-1]["code"].append(line)
    
    # Format sections
    problem_statement = "\n".join(problem_statement)
    formatted_snippets = []
    for snippet in snippets:
        formatted_snippets.append(
            f"File: {snippet['file']}\nCode:\n" + "\n".join(snippet['code'])
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
        max_new_tokens=1024,  # Increase token limit for better output
        temperature=0.2,  # Lower temperature for more deterministic output
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.15,
        batch_size=4  # Increase batch size to utilize more VRAM
    )[0]['generated_text']
    
    # Extract diff
    model_patch = extract_diff_from_response(response)
    
    # Remove the 'codebase/{instance_id}/' prefix
    model_patch = remove_codebase_prefix(model_patch, instance_id)
    
    return {
        "instance_id": instance_id,
        "model_patch": model_patch,
        "model_name_or_path": model_name_or_path
    }

# Process all instances
results = []
for instance_id in os.listdir(input_data_dir):
    instance_dir = os.path.join(input_data_dir, instance_id)
    if os.path.isdir(instance_dir):
        result = process_instance(instance_id)
        if result:
            results.append(result)
            print(f"Processed {instance_id}")

# Save results
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")