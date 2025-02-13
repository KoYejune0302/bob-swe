#!/bin/bash

# Function to run a Python script and print its status
run_script() {
    script_name=$1
    echo "Running $script_name..."
    if python3 $script_name; then
        echo "$script_name completed successfully."
    else
        echo "$script_name failed."
        exit 1
    fi
    echo "----------------------------------------"
}

# List of scripts to run
scripts=(
    "load_dataset.py"
    "load_codebase.py"
    "extract_input.py"
    "run_llm_swe_llama.py"
)

# Run each script in order
for script in "${scripts[@]}"; do
    run_script $script
done

echo "All scripts completed successfully."