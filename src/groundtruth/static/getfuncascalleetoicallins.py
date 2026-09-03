#!/usr/bin/env python3
"""Invert indirect-call mappings so each callee points back to its callsites."""

import os
import json
import argparse
import sys
import glob

def reverse_json_mapping(input_file, output_file):
    """
    Reverse the key-value mapping in a JSON file where all values are lists.
    
    For input like:
    {
        "call_addr1": ["func_addr1"],
        "call_addr2": ["func_addr1", "func_addr2"],
        "call_addr3": ["func_addr3"]
    }
    
    Output will be:
    {
        "func_addr1": ["call_addr1", "call_addr2"],
        "func_addr2": ["call_addr2"],
        "func_addr3": ["call_addr3"]
    }
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the input JSON file
        with open(input_file, 'r') as f:
            call_to_func_mapping = json.load(f)
        
        # Create reverse mapping
        func_to_calls = {}
        
        # Iterate through the original mapping
        for call_addr, func_addrs in call_to_func_mapping.items():
            # Ensure func_addrs is treated as a list
            if not isinstance(func_addrs, list):
                func_addrs = [func_addrs]
                
            # Process each function address in the list
            for func_addr in func_addrs:
                # If function address not in the mapping yet, add it
                if func_addr not in func_to_calls:
                    func_to_calls[func_addr] = []
                
                # Add the call address to the list for this function
                func_to_calls[func_addr].append(call_addr)
        
        # Save the reversed mapping to the output file
        with open(output_file, 'w') as f:
            json.dump(func_to_calls, f, indent=2)
            
        return True
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False

def process_folder(binary, sourceinfo):
    """
    Process all JSON files in the input folder and generate reversed mappings
    in the output folder.
    
    Args:
        input_folder (str): Path to folder containing input JSON files
        output_folder (str): Path to folder where output JSON files will be saved
    """
    
    # Find all *_icallinstocallee.json files in the input folder
    input_file = os.path.join(sourceinfo, f"{binary}_icallinstocallee.json")
    
    if not input_file:
        print(f"Error: No *_icallinstocallee.json files found in")
        return
        
    # Create the output filename
    output_file = os.path.join(sourceinfo, f"{binary}_funcascalleetoicallins.json")
    
    
    # Process the file
    if reverse_json_mapping(input_file, output_file):
        print(f"  Successfully generated {os.path.basename(output_file)}")
    
def main():
    
    if len(sys.argv) != 3:
        sys.exit(1)

    binary_name = sys.argv[1]
    sourceinfo = sys.argv[2]  # Folder containing the jmptable.json
    
    # Process the folder
    process_folder(binary_name, sourceinfo)

if __name__ == "__main__":
    main()
