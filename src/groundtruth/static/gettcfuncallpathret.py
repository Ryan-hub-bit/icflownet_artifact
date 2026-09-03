#!/usr/bin/env python3
"""Trace tail-call function chains and collect all reachable return blocks."""

import json
import os
import sys
import glob


def find_all_final_targets_with_returns(func_addr, direct_calls, indirect_calls, func_to_ret):
    """
    Find all final tail call targets and collect all return blocks along all possible call paths.
    
    Args:
        func_addr: Address of the function to analyze
        direct_calls: Dictionary mapping function addresses to direct tail call target (single string)
        indirect_calls: Dictionary mapping function addresses to indirect tail call targets (list)
        func_to_ret: Dictionary mapping function addresses to return blocks
    
    Returns:
        Set of all return blocks encountered along all tail call paths
    """
    # Set to store all return blocks found
    ret_blocks = set()
    
    # Set to track visited functions to avoid cycles
    visited = set()
    
    def dfs(current_func):
        # Skip if we've already visited this function
        if current_func in visited:
            return
        
        # Mark as visited
        visited.add(current_func)
        
        # Add any return blocks for the current function
        if current_func in func_to_ret:
            ret_blocks.update(func_to_ret[current_func])
        
        # Process direct tail call (single target)
        if current_func in direct_calls:
            direct_target = direct_calls[current_func]
            dfs(direct_target)
        
        # Process indirect tail calls (list of targets)
        if current_func in indirect_calls:
            for target in indirect_calls[current_func]:
                dfs(target)
    
    # Start the depth-first search from the given function address
    dfs(func_addr)
    
    return ret_blocks

def process_binary(binary_name, angrcfginfo, sourceinfo):
    """
    Process a single binary to create the mapping from tail call functions to final targets
    and all return blocks along the call path.

    Args:
        binary_name: Name of the binary
        folder_path: Path to the folder containing JSON files

    Returns:
        True if successful, False otherwise
    """
    # File paths
    direct_file = os.path.join(sourceinfo, f"{binary_name}_dtcfunctocallee.json")
    indirect_file = os.path.join(sourceinfo, f"{binary_name}_itcfunctocallee.json")
    func_to_ret_file = os.path.join(angrcfginfo, f"{binary_name}_functoret.json")
    output_ret_blocks_file = os.path.join(angrcfginfo, f"{binary_name}_tcfunctoallpathret.json")

    print(f"Processing binary: {binary_name}")
    print(f"  Direct tail call file: {direct_file}")
    print(f"  Indirect tail call file: {indirect_file}")
    print(f"  Function to return blocks file: {func_to_ret_file}")

    # Check if all files exist
    if not os.path.exists(direct_file):
        print(f"  Error: Direct tail call file not found: {direct_file}")

    if not os.path.exists(indirect_file):
        print(f"  Error: Indirect tail call file not found: {indirect_file}")

    if not os.path.exists(func_to_ret_file):
        print(f"  Error: Function to return blocks file not found: {func_to_ret_file}")

    # Load the JSON files
    try:
        with open(direct_file, 'r') as f:
            direct_calls = json.load(f)

        with open(indirect_file, 'r') as f:
            indirect_calls = json.load(f)

        with open(func_to_ret_file, 'r') as f:
            func_to_ret = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  Error: Invalid JSON in input file - {e}")
        return False

    print(f"  Loaded {len(direct_calls)} direct tail calls, {len(indirect_calls)} indirect tail calls, "
          f"and {len(func_to_ret)} function to return block mappings")

    # Get all function addresses from both call files
    all_funcs = set(direct_calls.keys()) | set(indirect_calls.keys())
    print(f"  Found {len(all_funcs)} unique function addresses")

    # Create the final target mapping and return block mapping
    return_block_mapping = {}

    for func_addr in all_funcs:
        ret_blocks = find_all_final_targets_with_returns(
            func_addr, direct_calls, indirect_calls, func_to_ret)

        # Store return blocks mapping
        return_block_mapping[func_addr] = list(ret_blocks)


    with open(output_ret_blocks_file, 'w') as f:
        json.dump(return_block_mapping, f, indent=2)
    print(f"  Return block mapping written to: {output_ret_blocks_file}")

    return True

def main():
    if len(sys.argv) != 4:
        sys.exit(1)
    binary_name = sys.argv[1]
    angrinfo_path = sys.argv[2]
    sourceinfo = sys.argv[3]
    process_binary(binary_name, angrinfo_path,sourceinfo)

if __name__ == "__main__":
    main()
