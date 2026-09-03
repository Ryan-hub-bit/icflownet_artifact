#!/usr/bin/env python3
"""Build indirect-call return-to-after-call mappings for static supervision."""

import os
import json
import glob
import sys



def process_json_files(binary, angrcfginfo,sourceinfo, output_folder):
    # Step 2: Read the JSON files
    icallinstocallee_path = os.path.join(sourceinfo, f"{binary}_icallinstocallee.json")
    functoret_path = os.path.join(angrcfginfo, f"{binary}_functoret.json")
    calltoaftercall_path = os.path.join(angrcfginfo, f"{binary}_calltoaftercall.json")
    
    with open(icallinstocallee_path, 'r') as f:
        icallinstocallee = json.load(f)
    
    with open(functoret_path, 'r') as f:
        functoret = json.load(f)
    
    with open(calltoaftercall_path, 'r') as f:
        calltoaftercall = json.load(f)
    
    # Step 3: Create the mapping
    idcrettoaftercall = {}
    
    # For each key in dcinstofunc.json
    for key, func_addrs in icallinstocallee.items():
        # Check if func_addr exists in functoret.json
        for func_addr in func_addrs:
          if func_addr in functoret:
              ret_addr = functoret[func_addr]
              # For each return address (value)
              if not ret_addr:
                continue
              if isinstance(ret_addr, list):
                  for addr in ret_addr:
                    if key in calltoaftercall:
                        if addr not in idcrettoaftercall:
                            idcrettoaftercall[addr]=[]
                        if calltoaftercall[key] not in idcrettoaftercall[addr]:
                          idcrettoaftercall[addr].append(calltoaftercall[key])
              else:
                print(f'{ret_addr} is not a list')
    
    output_path = os.path.join(output_folder, f"{binary}_idcreturntoaftercall.json")
    # Step 4: Save the mapping to the output file
    with open(output_path, 'w') as f:
        json.dump(idcrettoaftercall, f, indent=4)
    
    print(f"Mapping created and saved to {output_path}")

def main():
    if len(sys.argv) != 5:
        sys.exit(1)
        
    binary_name = sys.argv[1]
    angrcfginfo = sys.argv[2]  # Folder containing the jmptable.json
    sourceinfo = sys.argv[3]  # Folder containing the jmptable.json
    output_folder = sys.argv[4]  # Folder
    # Get folder paths from user or set default paths
    # Process the files
    process_json_files(binary_name, angrcfginfo, sourceinfo, output_folder)

if __name__ == "__main__":
    main()
