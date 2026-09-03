#!/usr/bin/env python3
"""Build direct-call return-to-after-call mappings from intermediate JSON files."""

import os
import json
import glob
import sys



def process_json_files(binary, folder, output_folder):
    """
    Process the JSON files to create the required mapping:
    1. Find binary name
    2. Read {binary}_dcinstofunc.json
    3. For each key-value pair, check if value exists in {binary}_functoret.json
    4. Find corresponding entries in {binary}_calltoaftercall.json
    5. Create final mapping and save to output file
    """
    
    # Step 2: Read the JSON files
    dcinstofunc_path = os.path.join(folder, f"{binary}_dcinstofunc.json")
    functoret_path = os.path.join(folder, f"{binary}_functoret.json")
    calltoaftercall_path = os.path.join(folder, f"{binary}_calltoaftercall.json")
    
    with open(dcinstofunc_path, 'r') as f:
        dcinstofunc = json.load(f)
    
    with open(functoret_path, 'r') as f:
        functoret = json.load(f)
    
    with open(calltoaftercall_path, 'r') as f:
        calltoaftercall = json.load(f)
    
    # Step 3: Create the mapping
    dcrettoaftercall = {}
    
    # For each key in dcinstofunc.json
    for key, func_addr in dcinstofunc.items():
        # Check if func_addr exists in functoret.json
        if func_addr in functoret:
            ret_addr = functoret[func_addr]
            # For each return address (value)
            if not ret_addr:
                continue
            if isinstance(ret_addr, list):
                for addr in ret_addr:
                    if key in calltoaftercall:
                        if addr not in dcrettoaftercall:
                            dcrettoaftercall[addr]=[]
                        dcrettoaftercall[addr].append(calltoaftercall[key])
            else:
                print(f'{ret_addr} is not a list')
    
    output_path = os.path.join(output_folder, f"{binary}_dcreturntoaftercall.json")
    # Step 4: Save the mapping to the output file
    with open(output_path, 'w') as f:
        json.dump(dcrettoaftercall, f, indent=4)
 

def main():

    
    if len(sys.argv) != 4:
        sys.exit(1)
        
    binary_name = sys.argv[1]  
    angrcfginfo = sys.argv[2]  
    output_folder = sys.argv[3]  

    # Get folder paths from user or set default paths
    # Process the files
    process_json_files(binary_name, angrcfginfo, output_folder)

if __name__ == "__main__":
    main()
