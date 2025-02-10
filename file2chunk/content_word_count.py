import json
import sys
import os

def analyze_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    word_counts = []
    lowercase_ids = []
    ids_over_300 = []
    ids_under_100 = []
    id_to_word_count = {}  # Dictionary to store id-word count pairs

    for item in data:
        # Skip items with type 'table'
        if item.get("type") == "table":
            continue

        content = item.get("content")
        if content:
            word_count = len(content.split())
            word_counts.append(word_count)
            item_id = item.get("id")
            id_to_word_count[item_id] = word_count
            
            if content[0].islower():
                lowercase_ids.append(item_id)
            if word_count > 300:
                ids_over_300.append(item_id)
            if word_count < 100:
                ids_under_100.append(item_id)

    if not word_counts:
        print("No valid 'content' fields found.")
        return

    average_word_count = sum(word_counts) / len(word_counts)
    max_word_count = max(word_counts)
    min_word_count = min(word_counts)

    # Find IDs with max and min word counts
    max_id = max(id_to_word_count.items(), key=lambda x: x[1])[0]
    min_id = min(id_to_word_count.items(), key=lambda x: x[1])[0]

    # Determine the output file path
    output_dir = os.path.dirname(file_path)
    output_file = os.path.join(output_dir, "word_count_stats.txt")

    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write(f"Average word count: {average_word_count}\n")
        out_file.write(f"Max word count: {max_word_count} (ID: {max_id})\n")
        out_file.write(f"Min word count: {min_word_count} (ID: {min_id})\n\n")

        if lowercase_ids:
            out_file.write("IDs with content starting with a lowercase letter:\n")
            out_file.write(", ".join(map(str, lowercase_ids)) + "\n\n")

        if ids_over_300:
            out_file.write("IDs with content word count over 300:\n")
            out_file.write(", ".join(map(str, ids_over_300)) + "\n\n")

        if ids_under_100:
            out_file.write("IDs with content word count under 100:\n")
            out_file.write(", ".join(map(str, ids_under_100)) + "\n")

    print(f"Statistics written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_json.py <path_to_json_file>")
    else:
        json_file_path = sys.argv[1]
        if os.path.exists(json_file_path):
            analyze_json(json_file_path)
        else:
            print(f"File not found: {json_file_path}")