"""Lift tail-call instruction-level mappings to function-level callee mappings."""

import json
import os
import sys

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {file_path}")
        return {}

def write_json_file(file_path, data):
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error writing to file {file_path}: {str(e)}")

def transform_data(input_data, mapping_data):
    result = {}
    for input_key, input_value in input_data.items():
        new_key = mapping_data.get(input_key, input_key)
        result[new_key] = input_value
    return result

def process_binary(binary_name, angrcfginfo, sourceinfo):
    dtc_input_path = os.path.join(sourceinfo, f"{binary_name}_dtcinstocallee.json")
    itc_input_path = os.path.join(sourceinfo, f"{binary_name}_itcinstofunc.json")
    jump_input_path = os.path.join(angrcfginfo, f"{binary_name}_jumpinstofunc.json")

    dtc_output_path = os.path.join(sourceinfo, f"{binary_name}_dtcfunctocallee.json")
    itc_output_path = os.path.join(sourceinfo, f"{binary_name}_itcfunctocallee.json")

    print(f"\nProcessing binary: {binary_name}")

    dtc_data = read_json_file(dtc_input_path)
    itc_data = read_json_file(itc_input_path)
    jump_data = read_json_file(jump_input_path)

    if not jump_data:
        print(f"Warning: Skipping {binary_name} due to missing or invalid jump mapping.")
        return

    dtc_transformed = transform_data(dtc_data, jump_data)
    itc_transformed = transform_data(itc_data, jump_data)

    write_json_file(dtc_output_path, dtc_transformed)
    write_json_file(itc_output_path, itc_transformed)

    print(f"  → Written: {os.path.basename(dtc_output_path)}")
    print(f"  → Written: {os.path.basename(itc_output_path)}")

def main(binary, angrcfginfo, sourceinfo):
    process_binary(binary, angrcfginfo, sourceinfo)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(1)
    binary_name = sys.argv[1]
    angrcfginfo = sys.argv[2]
    sourceinfo = sys.argv[3]
    main(binary_name, angrcfginfo, sourceinfo)
