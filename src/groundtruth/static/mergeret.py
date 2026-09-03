#!/usr/bin/env python3
"""Merge return-to-after-call mappings from direct, indirect, and tail-call flows."""

import json
import os
import sys
import glob
from collections import defaultdict

def merge_json_files(binary_name, folder_path, binary_path, map_file):
    """
    Merge three types of JSON files related to a binary name:
    - {binary}_dcreturntoaftercall.json
    - {binary}_idcreturntoaftercall.json
    - {binary}*rettoaftercallfortc.json
    
    Args:
        binary_name (str): The name of the binary
        folder_path (str): Path to the folder containing the files
    """
    # Define the file patterns to search for
    dc_pattern = os.path.join(folder_path, f"{binary_name}_dcreturntoaftercall.json")
    idc_pattern = os.path.join(folder_path, f"{binary_name}_idcreturntoaftercall.json")
    tc_pattern = os.path.join(folder_path, f"{binary_name}_rettoaftercallfortc.json")
    
    # Get the actual file paths
    dc_files = glob.glob(dc_pattern)
    idc_files = glob.glob(idc_pattern)
    tc_files = glob.glob(tc_pattern)
    
    # Check if files exist
    if not dc_files:
        print(f"Warning: No files found matching {dc_pattern}")
    if not idc_files:
        print(f"Warning: No files found matching {idc_pattern}")
    if not tc_files:
        print(f"Warning: No files found matching {tc_pattern}")
    
    if not (dc_files or idc_files or tc_files):
        print(f"Error: No files found to merge. Exiting.")
        return False
    
    # Initialize merged data container
    merged_data = defaultdict(dict)
    
    # Process DC files
    for dc_file in dc_files:
        try:
            with open(dc_file, 'r') as f:
                dc_data = json.load(f)
            
            for key, value in dc_data.items():
                # If the key doesn't exist yet, initialize it with the value
                if key not in merged_data:
                    merged_data[key] = value
                # Otherwise, extend the list with unique values
                else:
                    merged_data[key].extend([item for item in value if item not in merged_data[key]])
            
            print(f"Processed {dc_file}")
        except Exception as e:
            print(f"Error processing {dc_file}: {e}")
    
    # Process IDC files
    for idc_file in idc_files:
        try:
            with open(idc_file, 'r') as f:
                idc_data = json.load(f)
            
            for key, value in idc_data.items():
                # If the key doesn't exist yet, initialize it with the value
                if key not in merged_data:
                    merged_data[key] = value
                # Otherwise, extend the list with unique values
                else:
                    merged_data[key].extend([item for item in value if item not in merged_data[key]])
            
            print(f"Processed {idc_file}")
        except Exception as e:
            print(f"Error processing {idc_file}: {e}")
    
    # Process TC files
    for tc_file in tc_files:
        try:
            with open(tc_file, 'r') as f:
                tc_data = json.load(f)
            
            for key, value in tc_data.items():
                # If the key doesn't exist yet, initialize it with the value
                if key not in merged_data:
                    merged_data[key] = value
                # Otherwise, extend the list with unique values
                else:
                    merged_data[key].extend([item for item in value if item not in merged_data[key]])
            
            print(f"Processed {tc_file}")
        except Exception as e:
            print(f"Error processing {tc_file}: {e}")
    
    # Create the output file path
    output_file = os.path.join(folder_path, f"{binary_name}_ret.json")
          # Step 5: Open map_file and append the key-value pair
    map_data = {}
        # Check if map_file exists and load its content if it does
    if os.path.exists(map_file):
        with open(map_file, 'r') as f:
            try:
                map_data = json.load(f)
            except json.JSONDecodeError:
                # If file exists but is not valid JSON or is empty
                map_data = {}
    
    # Add or update the key-value pair
    map_data[output_file] = binary_path
    # Write the updated map data back to the file
    with open(map_file, 'w') as f:
        json.dump(map_data, f, indent=4)
    # Write the merged data to the output file
    try:
        with open(output_file, 'w') as f:
            json.dump(dict(merged_data), f, indent=4)
        
        print(f"Successfully created merged file: {output_file}")
        return True
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False
    
    
 
    
def main():
    # Check arguments
    if len(sys.argv) != 5:
        print("Usage: python merge_json_files.py <binary_name> <folder_path>")
        return
    
    binary_name = sys.argv[1]
    folder_path = sys.argv[2]
    binary_path = sys.argv[3]
    map_file = sys.argv[4]
    
    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        return
    
    # Merge the files
    merge_json_files(binary_name, folder_path,binary_path, map_file)

if __name__ == "__main__":
    main()
