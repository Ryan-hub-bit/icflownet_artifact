"""Repair jump-table source blocks so they align with angr basic-block addresses."""

import json
import os
import sys
from collections import defaultdict

def correct_jump_mappings(binary_name, angrinfo, sourceinfo, output_folder):
    # Find the jmptable.json file in folder1 to determine the binary name
    jmptable_path = os.path.join(sourceinfo, f"{binary_name}_jmptable.json")
    
    # Load the jmptable.json file
    try:
        with open(jmptable_path, 'r') as file:
            jmptable = json.load(file)
    except Exception as e:
        print(f"Error loading {jmptable_path}: {e}")
        return []
    
    # Define paths for the other files
    jumpbblist_path = os.path.join(angrinfo, f"{binary_name}_jumpbblist.json")
    bbtofunc_path = os.path.join(angrinfo, f"{binary_name}_bbtofunc.json")
    
    # Load the jumpbblist.json file
    try:
        with open(jumpbblist_path, 'r') as file:
            jumpbblist = json.load(file)
    except FileNotFoundError:
        print(f"Error: Could not find {jumpbblist_path}")
        return []
    except Exception as e:
        print(f"Error loading {jumpbblist_path}: {e}")
        return []
    
    # Load the bbtofunc.json file
    try:
        with open(bbtofunc_path, 'r') as file:
            bbtofunc = json.load(file)
    except FileNotFoundError:
        print(f"Error: Could not find {bbtofunc_path}")
        return []
    except Exception as e:
        print(f"Error loading {bbtofunc_path}: {e}")
        return []
    
    # Extract the jump basic blocks list
    jump_bbs = set(jumpbblist.get("jumpbb", []))
    
    # Check if all keys in jmptable are in jumpbblist
    missing_in_jumpbblist = []
    for key in jmptable.keys():
        if key not in jump_bbs:
            missing_in_jumpbblist.append(key)
    
    # Sort all addresses in bbtofunc for finding nearest lower address
    sorted_addresses = sorted([int(addr, 16) for addr in bbtofunc.keys()])
    
    # Create a corrected jmptable
    corrected_jmptable = jmptable.copy()
    corrections = {}
    
    for missing_key in missing_in_jumpbblist:
        missing_addr = int(missing_key, 16)
        # Find the closest lower address
        closest_lower = None
        for addr in sorted_addresses:
            if addr < missing_addr and (closest_lower is None or addr > closest_lower):
                closest_lower = addr
        
        if closest_lower is not None:
            closest_lower_hex = f"0x{closest_lower:x}"
            corrections[missing_key] = closest_lower_hex
            # Replace the key in the corrected jmptable
            if missing_key in corrected_jmptable:
                targets = corrected_jmptable.pop(missing_key)
                corrected_jmptable[closest_lower_hex] = targets
    
    # Save the corrected jmptable
    output_path = os.path.join(output_folder, f"{binary_name}_correctjumptable.json")
    try:
        with open(output_path, 'w') as file:
            json.dump(corrected_jmptable, file, indent=2)
        print(f"Corrected jump table saved to: {output_path}")
    except Exception as e:
        print(f"Error saving corrected jump table: {e}")
        return []
    
    return corrected_jmptable.keys()
    
    # Print corrections
    # print("\nCorrections made:")
    # for original, corrected in corrections.items():
    #     print(f"  {original} -> {corrected}")
    
    # print(f"\nFound {len(missing_in_jumpbblist)} keys in jmptable that are missing in jumpbblist:")
    # for key in missing_in_jumpbblist:
    #     print(f"  {key}")
    
    # # Additional statistics
    # print(f"\nStatistics:")
    # print(f"  Total jump sources in original jmptable: {len(jmptable)}")
    # print(f"  Total jump sources in corrected jmptable: {len(corrected_jmptable)}")
    # print(f"  Total corrections made: {len(corrections)}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python correct_jump_mappings.py <folder_with_jmptable> <folder_with_jumpbblist_and_bbtofunc> <output_folder>")
        sys.exit(1)
    binary_name = sys.argv[1] 
    angrinfo = sys.argv[2]
    sourceinfo = sys.argv[3]
    output_folder = sys.argv[4]  # Folder for the corrected jmptable
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    correct_jump_mappings(binary_name, angrinfo, sourceinfo, output_folder)
