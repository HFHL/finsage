import json
import sys
import os

def process_json(file_path):
    # Read the JSON data from the specified file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Filter out chunks with empty content
    processed_data = [chunk for chunk in data if 'content' in chunk and chunk['content'] != ""]
    
    # Create a new file name with the '_remove_empty' suffix
    base_name, ext = os.path.splitext(file_path)
    new_file_path = f"{base_name}_remove_empty{ext}"
    
    # Write the processed data to the new file
    with open(new_file_path, 'w', encoding='utf-8') as file:
        json.dump(processed_data, file, indent=2, ensure_ascii=False)
    
    print(f"Processed JSON saved to: {new_file_path}")

if __name__ == "__main__":
    # Check if the file path is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_json_file>")
        sys.exit(1)
    
    # Get the file path from the command-line arguments
    json_file_path = sys.argv[1]
    
    # Process the JSON file
    process_json(json_file_path)