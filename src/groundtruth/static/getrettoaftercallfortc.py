"""Map tail-call return sites to the after-call blocks that can follow them."""

import json
import os
import sys
from pathlib import Path

def process_json_files(binary_name, angrcfginfo, res):
    
    # File paths (joined with the input directory)
    func_to_callins_path = os.path.join(angrcfginfo, f"{binary_name}_funcascalleetoallcallins.json")
    call_to_aftercall_path = os.path.join(angrcfginfo, f"{binary_name}_calltoaftercall.json")
    tc_func_to_pathret_path = os.path.join(angrcfginfo, f"{binary_name}_tcfunctoallpathret.json")
    
    # Output path (joined with the output directory)
    output_path = os.path.join(res, f"{binary_name}_rettoaftercallfortc.json")
    
    # Load the JSON files
    try:
        with open(func_to_callins_path, 'r') as f:
            func_to_callins = json.load(f)
        
        with open(call_to_aftercall_path, 'r') as f:
            call_to_aftercall = json.load(f)
        
        with open(tc_func_to_pathret_path, 'r') as f:
            tc_func_to_pathret = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the required JSON files: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in one of the files: {e}")
        sys.exit(1)
    
    print(f"Loaded files successfully.")
    print(f"Processing data...")
    
    # Create the ret to after-callins mapping
    ret_to_aftercallins = {}
    
    # Process each function in the func_to_callins file
    for func_addr, callins in func_to_callins.items():
        # Check if this function exists in tc_func_to_pathret
        if func_addr in tc_func_to_pathret:
            ret_addresses = tc_func_to_pathret[func_addr]
            
            # For each call-in, find the after-call-ins
            all_aftercallins = set()
            for callin in callins:
                if callin in call_to_aftercall:
                    aftercallins = call_to_aftercall[callin]
                    all_aftercallins.add(aftercallins)
            
            # Map each ret address to the set of after-callins
            for ret_addr in ret_addresses:
                if ret_addr not in ret_to_aftercallins:
                    ret_to_aftercallins[ret_addr] = []
                # Add all after-callins to this return address
                for aftercall in all_aftercallins:
                    if aftercall not in ret_to_aftercallins[ret_addr]:
                        # print(aftercall)
                        ret_to_aftercallins[ret_addr].append(aftercall)
    
    # Ensure uniqueness in all lists
    for ret_addr in ret_to_aftercallins:
        # Convert to set and back to list to ensure uniqueness
        ret_to_aftercallins[ret_addr] = list(set(ret_to_aftercallins[ret_addr]))
        # Sort for consistent output (optional)
        ret_to_aftercallins[ret_addr].sort()
    
    # Save the output file
    with open(output_path, 'w') as f:
        json.dump(ret_to_aftercallins, f, indent=2)
    
    print(f"Successfully created {output_path}")
    print(f"Output file: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_directory_path> <output_directory_path>")
        sys.exit(1)
    
    binary_name = sys.argv[1]
    angrcfginfo = sys.argv[2]
    res = sys.argv[3]
    process_json_files(binary_name, angrcfginfo, res)
