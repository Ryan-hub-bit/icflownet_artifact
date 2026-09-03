"""Convert indirect tail-call mappings from instruction sites to basic-block keys."""

import json
import os
import sys

def process_itc_files(binary_name, angrcfginfo, sourceinfo, output_folder):
    # Find the itcinstofunc.json file in folder1 to determine the binary name
    itcinstofunc_path = os.path.join(sourceinfo, f"{binary_name}_itcinstofunc.json")
    
    # Load the itcinstofunc.json file
    try:
        with open(itcinstofunc_path, 'r') as file:
            itcinstofunc = json.load(file)
    except Exception as e:
        print(f"Error loading {itcinstofunc_path}: {e}")
        return []
    
    # Define path for the callinstobb.json file
    jumpinstobb_path = os.path.join(angrcfginfo, f"{binary_name}_jumpinstobb.json")
    instofunc_path = os.path.join(angrcfginfo,f"{binary_name}_instofunc.json")
    
    # Load the callinstobb.json file
    try:
        with open(jumpinstobb_path, 'r') as file:
            jumpinstobb = json.load(file)
    except FileNotFoundError:
        print(f"Error: Could not find {jumpinstobb_path}")
        return []
    except Exception as e:
        print(f"Error loading {jumpinstobb}: {e}")
        return []
    
    try:
        with open(instofunc_path, 'r') as file:
            instofunc = json.load(file)
    except FileNotFoundError:
        print(f"Error: Could not find {instofunc_path}")
        return []
    except Exception as e:
        print(f"Error loading {instofunc_path}: {e}")
        return []
    # Create the itcbbtofunc.json by replacing keys
    itcbbtofunc = {}
    missing_keys = []
   
    for inst_addr, func_addrs in itcinstofunc.items():
        if inst_addr in jumpinstobb:
            # Replace instruction address with basic block address
            bb_addr = jumpinstobb[inst_addr]
            correct_func_addrs = []
            for func_addr in func_addrs:
                if func_addr in instofunc:
                    correct_func_addrs.append(instofunc[func_addr])
                    #print(f"original_function addrss: {func_addr}, callee_addr:{instofunc[func_addr]}\n")
                else:
                    correct_func_addrs.append(func_addr)
            assert len(correct_func_addrs) == len(func_addrs), f"Lists have different lengths: {len(correct_func_addrs)} vs {len(func_addrs)}"
            itcbbtofunc[bb_addr] = correct_func_addrs
                
            # for issue sometimes that will insert a endbr64 when -fcf-protection
            # if func_addr in instofunc:
            #     callee_addr = instofunc[func_addr]
            #     itcbbtofunc[bb_addr] = callee_addr
            #     print(f"original_function addrss: {func_addr}, callee_addr:{callee_addr}\n")
            # else:
            #     itcbbtofunc[bb_addr] = func_addr

        else:
            # Record missing keys
            missing_keys.append(inst_addr)
            # Keep the original mapping
            itcbbtofunc[inst_addr] = func_addrs
    
    # Save the output file
    output_path = os.path.join(output_folder, f"{binary_name}_itcbbtofunc.json")
    try:
        with open(output_path, 'w') as file:
            json.dump(itcbbtofunc, file, indent=2)
        print(f"Output saved to: {output_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        return []
    
    
    
    # Print statistics and missing keys
    # print(f"\nStatistics:")
    # print(f"  Total entries in itcinstofunc: {len(itcinstofunc)}")
    # print(f"  Total entries in jumpinstobb: {len(jumpinstobb)}")
    # print(f"  Total entries in itcbbtofunc: {len(itcbbtofunc)}")
    
    if missing_keys:
        print(f"\nFound {len(missing_keys)} keys in itcinstofunc that are missing in callinstobb:")
        for key in missing_keys[:20]:  # Limit output to first 20 for readability
            print(f"  {key}")
        if len(missing_keys) > 20:
            print(f"  ... and {len(missing_keys) - 20} more")
    else:
        print("\nAll keys in itcinstofunc are present in callinstobb. ✓")

    return itcbbtofunc.keys()
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python process_itc_files.py <folder_with_itcinstofunc> <folder_with_callinstobb> <output_folder>")
        sys.exit(1)
    binary_name = sys.argv[1]   
    angrcfginfo = sys.argv[2]  # Folder containing callinstobb.json
    sourceinfo = sys.argv[3]  # Folder containing the itcinstofunc.json
    output_folder = sys.argv[4]  # Folder for the output
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    process_itc_files(binary_name, angrcfginfo, sourceinfo, output_folder)
