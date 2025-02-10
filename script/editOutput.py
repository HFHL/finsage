import os

def replace_string_in_files(directory):
    old_string = "--- Retrieved Chunks by EnsembleRetriever without HyDE(Up to 30 chunks)---"
    new_string = "--- Retrieved Chunks by EnsembleRetriever with HyDE(Up to 60 chunks)---"
    
    # Walk through all files in the directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            
            # Read the file content
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Replace the string
                if old_string in content:
                    content = content.replace(old_string, new_string)
                    
                    # Write back to the file
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(content)
                    print(f"Processed: {filepath}")
                
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")

# Directory path
directory = "/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/QAwithChunks/results_unfiltered/55_hyde"

# Run the replacement
replace_string_in_files(directory)
