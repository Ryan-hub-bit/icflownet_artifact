#!/usr/bin/env python3
"""Combine direct-call and indirect-call callee-to-callsite lookup files."""

import os
import json
import argparse
import sys
import glob

def combine_json_files(dc_file, ic_file, output_file):
    """
    Combine two JSON files based on their keys.
    
    Args:
        dc_file (str): Path to the direct call JSON file
        ic_file (str): Path to the indirect call JSON file
        output_file (str): Path to the output combined JSON file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the direct call JSON file
        with open(dc_file, 'r') as f:
            dc_mapping = json.load(f)
        
        # Load the indirect call JSON file
        with open(ic_file, 'r') as f:
            ic_mapping = json.load(f)
        
        # Create a combined mapping
        combined_mapping = {}
        
        # Add all entries from direct call mapping
        for func_addr, call_addrs in dc_mapping.items():
            combined_mapping[func_addr] = call_addrs.copy()
        
        # Add or merge entries from indirect call mapping
        for func_addr, call_addrs in ic_mapping.items():
            if func_addr in combined_mapping:
                # Function already exists in the combined mapping, add the new call addresses
                combined_mapping[func_addr].extend(call_addrs)
            else:
                # Function doesn't exist yet, add it with its call addresses
                combined_mapping[func_addr] = call_addrs.copy()
        
        # Save the combined mapping to the output file
        with open(output_file, 'w') as f:
            json.dump(combined_mapping, f, indent=2)
            
        return True
        
    except Exception as e:
        print(f"Error combining files: {str(e)}")
        return False

def process_binaries(binary, angrcfginfo, sourceinfo):
    """
    Process all matching binary files in the two input folders and generate
    combined JSON files in the output folder.
    
    Args:
        dc_folder (str): Path to folder containing *_funcascalleetodcallins.json files
        ic_folder (str): Path to folder containing *_funcascalleetoicallins.json files
        output_folder (str): Path to folder where combined JSON files will be saved
    """
    
    # Get all binary names from direct call folder
    dc_pattern = os.path.join(angrcfginfo, "*_funcascalleetodcallins.json")
    dc_files = glob.glob(dc_pattern)
    
    # Extract binary names from direct call files
    dc_binaries = {}
    for dc_file in dc_files:
        dc_binaries[binary] = dc_file
    
    # Get all binary names from indirect call folder
    ic_pattern = os.path.join(sourceinfo, "*_funcascalleetoicallins.json")
    ic_files = glob.glob(ic_pattern)
    
    # Extract binary names from indirect call files
    ic_binaries = {}
    for ic_file in ic_files:
        ic_binaries[binary] = ic_file
    
    # Find common binary names
    common_binaries = set(dc_binaries.keys()).intersection(set(ic_binaries.keys()))
    
    if not common_binaries:
        print("Error: No matching binary names found between the two folders")
        return
    
    print(f"Found {len(common_binaries)} matching binaries to process")
    

    dc_file = dc_binaries[binary]
    ic_file = ic_binaries[binary]
    output_file = os.path.join(angrcfginfo, f"{binary}_funcascalleetoallcallins.json")
    
    if combine_json_files(dc_file, ic_file, output_file):
        print(f"  Successfully combined files for {binary}")

def main():
    if len(sys.argv) != 4:
        sys.exit(1)
        
    binary_name = sys.argv[1]
    angrcfginfo = sys.argv[2]
    sourceinfo = sys.argv[3]  # Folder containing the jmptable.json

    
    # Process the folders
    process_binaries(binary_name, angrcfginfo, sourceinfo)

if __name__ == "__main__":
    main()
