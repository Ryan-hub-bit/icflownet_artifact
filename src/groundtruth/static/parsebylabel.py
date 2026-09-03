#!/usr/bin/env python3
"""Parse label metadata and emit source/target JSON files for GT generation."""

# parsebyLabel  parse the indirect jump/call pairs according to their lable by indirect jump type(indirect call ret, jump table, tail call)

import re
import sys
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import json

@dataclass
class SourceInfo:
    jumptable_id: Optional[int]
    tailcall_id: Optional[int]
    callsite_id: Optional[int]
    type_id: Optional[str]
    dtailcall_id: Optional[int]

@dataclass
class DestInfo:
    jumptable_source_id: Optional[int]
    jumptable_entry_id: Optional[int]
    return_id: Optional[int]
    function_entry_id: Optional[int]  # New field after return_id
    function_hash: Optional[str]
    type_id: Optional[str]

@dataclass
class Label:
    address: str
    symbol_type: str
    section: str
    size: str
    label_text: str
    module: str
    is_source: bool
    source_info: Optional[SourceInfo] = None
    dest_info: Optional[DestInfo] = None

def format_address(address: str) -> str:
    """Format address to include 0x prefix."""
    # Remove any existing leading zeros while preserving the actual address
    addr_without_leading_zeros = address.lstrip('0')
    # Add back necessary zeros to maintain length and add 0x prefix
    return f"0x{addr_without_leading_zeros}"

class LabelParser:
    def __init__(self, binary_path: str, gt_path: str):
        self.labels_by_address: Dict[str, List[Label]] = defaultdict(list)
        self.binary_path = binary_path
        self.binary_name = os.path.basename(binary_path)
        self.gt_path = gt_path

        # Basic relationships
        self.jumptable_to_address: Dict[int, str] = {}
        self.tailcall_to_address: Dict[int, str] = {}
        self.tailcall_to_typeid: Dict[int, str] = {}
        self.callsite_to_address: Dict[int, str] = {}
        self.callsite_to_typeid: Dict[int, str] = {}
        self.jumptable_entry_to_address: Dict[int, Set[str]] = defaultdict(set)
        self.return_to_typeid: Dict[int, str] = {}

        # Add dtailcall tracking
        self.dtailcall_to_address: Dict[int, str] = {}
        self.module_dtailcall_mapping = defaultdict(dict)

        # Module relationships
        self.module_to_source_info: Dict[str, List[SourceInfo]] = defaultdict(list)
        self.module_to_dest_info: Dict[str, List[DestInfo]] = defaultdict(list)

        # Store raw source and destination info by module
        self.module_sources: Dict[str, List[Tuple[str, SourceInfo]]] = defaultdict(list)
        self.module_destinations: Dict[str, List[Tuple[str, DestInfo]]] = defaultdict(list)

        self.module_jumptable_mapping: Dict[str, Dict[int, Tuple[str, List[str]]]] = defaultdict(dict)
        self.module_tailcall_mapping = defaultdict(dict)
        self.module_callsite_mapping = defaultdict(dict)

    def parse_source_info(self, source_text: str) -> Optional[SourceInfo]:
        """Parse source label to extract IDs."""
        parts = source_text.split('-')
        if len(parts) < 5:
            return None

        try:
            jumptable_id = int(parts[1]) if parts[1] != '0' else None
            tailcall_id = int(parts[2]) if parts[2] != '0' else None
            callsite_id = int(parts[3]) if parts[3] != '0' else None
            type_id = str(parts[4]) if parts[4] != '0' else None
            dtailcall_id = int(parts[5]) if len(parts) > 5 and parts[5] != '0' else None

            return SourceInfo(
                jumptable_id=jumptable_id,
                tailcall_id=tailcall_id,
                callsite_id=callsite_id,
                type_id=type_id,
                dtailcall_id=dtailcall_id
            )
        except (IndexError, ValueError):
            return None

    def parse_dest_info(self, dest_text: str) -> Optional[DestInfo]:
        """Parse destination label to extract IDs."""
        parts = dest_text.split('-')
        if len(parts) < 6:  # Ensure we have enough parts
            return None

        try:
            jumptable_source_id = int(parts[0]) if parts[0] != '0' else None
            jumptable_entry_id = int(parts[1]) if parts[1] != '0' else None
            return_id = int(parts[2]) if parts[2] != '0' else None
            function_entry_id = int(parts[3]) if parts[3] != '0' else None  # New field
            function_hash = parts[-2] if len(parts) >= 2 else None
            type_id = parts[-1] if len(parts) >= 1 else None

            return DestInfo(
                jumptable_source_id=jumptable_source_id,
                jumptable_entry_id=jumptable_entry_id,
                return_id=return_id,
                function_entry_id=function_entry_id,
                function_hash=function_hash,
                type_id=type_id
            )
        except (IndexError, ValueError):
            return None

    def parse_line(self, line: str) -> Optional[Tuple[Label, Label]]:
        """Parse a line into source and destination labels."""
        parts = line.strip().split(None, 4)
        if len(parts) != 5:
            return None

        address, symbol_type, section_info, size, full_label = parts
        # Format address with 0x prefix
        formatted_address = format_address(address)

        label_parts = full_label.split('-t-')
        if len(label_parts) != 2:
            return None

        source_text, dest_text = label_parts
        module = source_text.split('-')[0]

        source_info = self.parse_source_info(source_text)
        dest_info = self.parse_dest_info(dest_text)

        if source_info:
            self.module_to_source_info[module].append(source_info)
            self.module_sources[module].append((formatted_address, source_info))

            if source_info.jumptable_id is not None:
                self.jumptable_to_address[source_info.jumptable_id] = formatted_address
            if source_info.tailcall_id is not None:
                self.tailcall_to_address[source_info.tailcall_id] = formatted_address
                if source_info.type_id is not None:
                    self.tailcall_to_typeid[source_info.tailcall_id] = source_info.type_id
            if source_info.callsite_id is not None:
                self.callsite_to_address[source_info.callsite_id] = formatted_address
                if source_info.type_id is not None:
                    self.callsite_to_typeid[source_info.callsite_id] = source_info.type_id
            # Track dtailcall_id
            if source_info.dtailcall_id is not None:
                self.dtailcall_to_address[source_info.dtailcall_id] = formatted_address

        if dest_info:
            self.module_to_dest_info[module].append(dest_info)
            self.module_destinations[module].append((formatted_address, dest_info))

            if dest_info.jumptable_source_id is not None and dest_info.jumptable_entry_id is not None:
                entry_key = dest_info.jumptable_source_id
                self.jumptable_entry_to_address[entry_key].add(f"{dest_info.jumptable_entry_id}-{formatted_address}")

        source_label = Label(
            address=formatted_address,
            symbol_type=symbol_type.strip(),
            section=section_info.strip(),
            size=size,
            label_text=source_text,
            module=module,
            is_source=True,
            source_info=source_info
        )

        dest_label = Label(
            address=formatted_address,
            symbol_type=symbol_type.strip(),
            section=section_info.strip(),
            size=size,
            label_text=dest_text,
            module=module,
            is_source=False,
            dest_info=dest_info
        )

        return source_label, dest_label

    def process_jumptable_mappings(self):
        """Process jump table mappings after all lines have been parsed."""
        for module in self.module_sources.keys():
            # First collect all jump table sources
            jumptable_sources = {}
            for address, source_info in self.module_sources[module]:
                if source_info.jumptable_id is not None:
                    jumptable_sources[source_info.jumptable_id] = (address, [])

            # Then process all destinations for this module
            for address, dest_info in self.module_destinations[module]:
                if (dest_info.jumptable_source_id is not None and
                    dest_info.jumptable_entry_id is not None and
                    dest_info.jumptable_source_id in jumptable_sources):

                    source_addr, entries = jumptable_sources[dest_info.jumptable_source_id]
                    while len(entries) < dest_info.jumptable_entry_id:
                        entries.append(None)
                    entries[dest_info.jumptable_entry_id - 1] = address
                    jumptable_sources[dest_info.jumptable_source_id] = (source_addr, entries)

            # Clean up None entries and store final mappings
            for jmp_id, (source_addr, entries) in jumptable_sources.items():
                while entries and entries[-1] is None:
                    entries.pop()
                self.module_jumptable_mapping[module][jmp_id] = (source_addr, entries)

    def process_tailcall_mappings(self):
        """Process tail call mappings after all lines have been parsed."""
        self.module_tailcall_mapping = defaultdict(dict)

        # First, collect all tailcall sources across all modules
        all_tailcall_sources = {}
        for module in self.module_sources.keys():
            for address, source_info in self.module_sources[module]:
                if source_info.tailcall_id is not None and source_info.type_id is not None:
                    # Use a combined key of module and tailcall_id to keep them unique
                    key = (module, source_info.tailcall_id)
                    all_tailcall_sources[key] = (address, source_info.type_id, [])

        if not all_tailcall_sources:
            return

        # Then, look for matching destinations across all modules
        for dest_module in self.module_destinations.keys():
            for address, dest_info in self.module_destinations[dest_module]:
                # Check for both conditions: function_entry_id=1 and matching type_id
                if dest_info.function_entry_id == 1 and dest_info.type_id:
                    # Match across all tailcall sources regardless of module
                    for (src_module, tc_id), (src_addr, type_id, dest_addrs) in all_tailcall_sources.items():
                        if dest_info.type_id == str(type_id):
                            dest_addrs.append(address)
                            all_tailcall_sources[(src_module, tc_id)] = (src_addr, type_id, dest_addrs)

        # Finally, store the results back in module_tailcall_mapping
        for (module, tc_id), (src_addr, type_id, dest_addrs) in all_tailcall_sources.items():
            if dest_addrs:
                self.module_tailcall_mapping[module][tc_id] = (src_addr, type_id, dest_addrs)

    def process_callsite_mappings(self):
        """Process callsite mappings after all lines have been parsed."""
        self.module_callsite_mapping = defaultdict(dict)

        # First, collect all callsite sources across all modules
        all_callsite_sources = {}
        for module in self.module_sources.keys():
            for address, source_info in self.module_sources[module]:
                if source_info.callsite_id is not None and source_info.type_id is not None:
                    # Use a combined key of module and callsite_id to keep them unique
                    key = (module, source_info.callsite_id)
                    all_callsite_sources[key] = (address, source_info.type_id, [])

        if not all_callsite_sources:
            return

        # Then, look for matching destinations across all modules
        for dest_module in self.module_destinations.keys():
            for address, dest_info in self.module_destinations[dest_module]:
                if dest_info.return_id and dest_info.type_id:  # If destination has a type ID
                    # Match across all callsite sources regardless of module
                    for (src_module, cs_id), (src_addr, type_id, dest_addrs) in all_callsite_sources.items():
                        if dest_info.type_id == str(type_id):  # Compare type IDs
                            dest_addrs.append(address)
                            all_callsite_sources[(src_module, cs_id)] = (src_addr, type_id, dest_addrs)

        # Finally, store the results back in module_callsite_mapping
        for (module, cs_id), (src_addr, type_id, dest_addrs) in all_callsite_sources.items():
            if dest_addrs:  # Only store if we found matching destinations
                self.module_callsite_mapping[module][cs_id] = (src_addr, type_id, dest_addrs)

    def process_callsite_to_callee_mappings(self):
        """Process callsite to callee function mappings after all lines have been parsed."""
        self.module_callsite_to_callee_mapping = defaultdict(dict)

        # First, collect all callsite sources across all modules
        all_callsite_sources = {}
        for module in self.module_sources.keys():
            for address, source_info in self.module_sources[module]:
                if source_info.callsite_id is not None and source_info.type_id is not None:
                    # Use a combined key of module and callsite_id to keep them unique
                    key = (module, source_info.callsite_id)
                    all_callsite_sources[key] = (address, source_info.type_id, [])

        if not all_callsite_sources:
            return

        # Find matching callees (function entries with function_entry_id=1) across all modules
        for dest_module in self.module_destinations.keys():
            for address, dest_info in self.module_destinations[dest_module]:
                if dest_info.function_entry_id == 1 and dest_info.type_id:  # Function entry with type ID
                    # Match across all callsite sources regardless of module
                    for (src_module, cs_id), (src_addr, type_id, dest_addrs) in all_callsite_sources.items():
                        if dest_info.type_id == str(type_id):  # Compare type IDs
                            dest_addrs.append(address)
                            all_callsite_sources[(src_module, cs_id)] = (src_addr, type_id, dest_addrs)

        # Store the final mappings for callsites that have matching callees
        for (module, cs_id), (src_addr, type_id, callee_addrs) in all_callsite_sources.items():
            if callee_addrs:  # Only store if we found matching callees
                self.module_callsite_to_callee_mapping[module][cs_id] = (src_addr, type_id, callee_addrs)

    def process_dtailcall_addresses(self):
        """Collect all dtailcall addresses."""
        dtailcall_addresses = {}

        for module in self.module_sources.keys():
            for address, source_info in self.module_sources[module]:
                if source_info.dtailcall_id is not None:
                    # Format the address with 0x prefix if needed
                    formatted_address = address if address.startswith('0x') else format_address(address)

                    # Add to the module-specific dtailcall mapping
                    if source_info.dtailcall_id not in self.module_dtailcall_mapping[module]:
                        self.module_dtailcall_mapping[module][source_info.dtailcall_id] = formatted_address

                    # Add to the global dtailcall addresses dict
                    if formatted_address not in dtailcall_addresses:
                        dtailcall_addresses[formatted_address] = source_info.dtailcall_id

        return dtailcall_addresses

    def write_mapping_jsons(self):
        """Write jump table, callsite, tail call, and dtailcall mappings to separate JSON files with binary name prefix."""

        # Process and write jump table mappings
        jumptable_mappings = {}
        for module in sorted(self.module_jumptable_mapping.keys()):
            for jumptable_id, (source_addr, entry_addrs) in sorted(self.module_jumptable_mapping[module].items()):
                valid_entries = [addr for addr in entry_addrs if addr]
                if valid_entries:
                    formatted_source = format_address(source_addr) if not source_addr.startswith('0x') else source_addr
                    if formatted_source not in jumptable_mappings:
                        jumptable_mappings[formatted_source] = []
                    jumptable_mappings[formatted_source].extend(
                        [format_address(addr) if not addr.startswith('0x') else addr for addr in valid_entries]
                    )

        # print(self.gt_path)
        # print(self.binary_name)
        with open(f'{self.gt_path}/{self.binary_name}_jmptable.json', 'w') as f:
            json.dump(jumptable_mappings, f, indent=2)

        # Process and write tail call mappings
        tailcall_mappings = {}
        for module in sorted(self.module_tailcall_mapping.keys()):
            for tc_id, (source_addr, type_id, dest_addrs) in sorted(self.module_tailcall_mapping[module].items()):
                if dest_addrs:
                    formatted_source = format_address(source_addr) if not source_addr.startswith('0x') else source_addr
                    if formatted_source not in tailcall_mappings:
                        tailcall_mappings[formatted_source] = []
                    tailcall_mappings[formatted_source].extend(
                        [format_address(addr) if not addr.startswith('0x') else addr for addr in dest_addrs]
                    )

        with open(f'{self.gt_path}/{self.binary_name}_itcinstofunc.json', 'w') as f:
            json.dump(tailcall_mappings, f, indent=2)

        # Process and write dtailcall address mappings
        dtailcall_addresses = {}
        for module in sorted(self.module_sources.keys()):
            for address, source_info in self.module_sources[module]:
                if source_info.dtailcall_id is not None:
                    formatted_address = format_address(address) if not address.startswith('0x') else address
                    dtailcall_addresses[formatted_address] = source_info.dtailcall_id

        # Format the dtailcall addresses as a simple list for the output
        dtailcall_address_list = list(dtailcall_addresses.keys())
        dtailcall_output = {
            "dtailcall_addresses": dtailcall_address_list
        }

        with open(f'{self.gt_path}/{self.binary_name}_dtcinsaddr.json', 'w') as f:
            json.dump(dtailcall_output, f, indent=2)

        # Process and write callsite to callee mappings
        callsite_to_callee_mappings = {}
        for module in sorted(self.module_callsite_to_callee_mapping.keys()):
            for cs_id, (source_addr, type_id, callee_addrs) in sorted(self.module_callsite_to_callee_mapping[module].items()):
                if callee_addrs:
                    formatted_source = format_address(source_addr) if not source_addr.startswith('0x') else source_addr
                    if formatted_source not in callsite_to_callee_mappings:
                        callsite_to_callee_mappings[formatted_source] = []
                    callsite_to_callee_mappings[formatted_source].extend(
                        [format_address(addr) if not addr.startswith('0x') else addr for addr in callee_addrs]
                    )

        with open(f'{self.gt_path}/{self.binary_name}_icallinstocallee.json', 'w') as f:
            json.dump(callsite_to_callee_mappings, f, indent=2)

    def process_binary(self):
        """Run objdump and process its output."""
        try:
            completed = subprocess.run(
                ["objdump", "-t", self.binary_path],
                check=True,
                capture_output=True,
                text=True,
            )
            for line in completed.stdout.splitlines():
                if "t-" not in line:
                    continue
                result = self.parse_line(line)
                if result:
                    source_label, dest_label = result
                    self.labels_by_address[source_label.address].extend([source_label, dest_label])

            # Process mappings after all lines have been parsed
            self.process_jumptable_mappings()
            self.process_tailcall_mappings()
            self.process_callsite_mappings()
            self.process_callsite_to_callee_mappings()
            self.process_dtailcall_addresses()
            return True
        except subprocess.CalledProcessError as e:
            return False
        except FileNotFoundError:
            print("Error: objdump command not found. Please ensure binutils is installed.")
            sys.exit(1)

    def print_summary(self):
        """Print complete summary."""
        print("\n=== Label Summary ===")
        print(f"Binary: {self.binary_path}")

        # Count dtailcall addresses
        dtailcall_count = 0
        for module in self.module_sources.keys():
            for _, source_info in self.module_sources[module]:
                if source_info.dtailcall_id is not None:
                    dtailcall_count += 1

        print(f"Total dtailcall_id addresses found: {dtailcall_count}")


def is_valid_binary(file_path):
    """Check if a file is a valid binary executable."""
    try:
        # Check file exists and is readable
        if not os.path.isfile(file_path) or not os.access(file_path, os.R_OK):
            return False
        
        # Try to read the first few bytes to check for executable headers
        with open(file_path, 'rb') as f:
            header = f.read(4)
            
        # Check for ELF header (Linux/Unix executables)
        if header.startswith(b'\x7fELF'):
            return True
            
        # Check for MZ header (Windows executables)
        if header.startswith(b'MZ'):
            return True
            
        # Check for Mach-O header (macOS executables)
        if header in [b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf', b'\xca\xfe\xba\xbe', b'\xce\xfa\xed\xfe', b'\xcf\xfa\xed\xfe']:
            return True
            
        # Additional checks could be added for other binary formats
        
        return False
    except Exception as e:
        print(f"Error checking if {file_path} is a binary: {str(e)}")
        return False
def main():
    if len(sys.argv) != 3:
        print("Usage: python label_parser.py <path_to_binary_folder> <gt_path>")
        sys.exit(1)

    binary_path = sys.argv[1]
    gt_path = sys.argv[2]

    # Create output directory if it doesn't exist
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    if not is_valid_binary(binary_path):
        sys.exit(1)
    parser = LabelParser(binary_path, gt_path)
    if not parser.process_binary():
        print("No output from process_binary. Exiting.")
        sys.exit(1)  # Exit with code 0 (success, but no data)
    # parser.print_summary()
    parser.write_mapping_jsons()

if __name__ == '__main__':
    main()
