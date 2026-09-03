#!/usr/bin/env python3
"""Convert indirect-call targets from instruction addresses to basic-block keys."""

import os
import json
import glob
import sys


def process_icall(binary, angrcfginfo, sourceinfo, output_folder):
    """
    Process the JSON files:
    1. Find binary name
    2. Read {binary}_icallinstocallee.json from folder1
    3. Read {binary}_callinstobb.json from folder2
    4. For each key in icallinstocallee.json, find its value in callinstobb.json with the same key
    5. Replace the value with the original key and create {binary}_callbbtocallee.json in folder3
    """
    # Step 2: Read the JSON files
    # TODO need to verify if the callee is the function address
    # in some case the label will be marked to the real function address
    # IBT
    # clang -fcf-protection=branch   # Just IBT 
    # https://claude.ai/share/f6c5d85e-a12e-494b-a700-5b632b3ab9f7
    # .text:0000000000007090
    #.text:0000000000007090 ; PyObject *__cdecl PyInit_decoder()
    #.text:0000000000007090                 public PyInit_decoder
    #.text:0000000000007090 PyInit_decoder  proc near               ; DATA XREF: LOAD:00000000000012B8↑o
    #.text:0000000000007090 ; __unwind {
    #.text:0000000000007090                 endbr64
    #.text:0000000000007094
    #.text:0000000000007094 xpra_codecs_jpeg_decoder_c_0_0_0_0_0_t_0_0_1_1_529260aac7f9137c_d44c56a03ce963e2:
    #.text:0000000000007094                 push    rbp
    #.text:0000000000007095                 mov     rbp, rsp
    #.text:0000000000007098                 lea     rdi, __pyx_moduledef
    #.text:000000000000709F                 pop     rbp
    #.text:00000000000070A0                 jmp     cs:PyModuleDef_Init_ptr
    #.text:00000000000070A0 ; } // starts at 7090
    #.text:00000000000070A0 PyInit_decoder  endp
    icallinstocallee_path = os.path.join(sourceinfo, f"{binary}_icallinstocallee.json")
    callinstobb_path = os.path.join(angrcfginfo, f"{binary}_callinstobb.json")
    output_path = os.path.join(output_folder, f"{binary}_icallbbtocallee.json")
    instofunc_path = os.path.join(angrcfginfo, f"{binary}_instofunc.json")
    
    with open(icallinstocallee_path, 'r') as f:
        icallinstocallee = json.load(f)
    
    with open(callinstobb_path, 'r') as f:
        callinstobb = json.load(f)

    with open(instofunc_path, 'r') as f:
        instofunc = json.load(f)
    
    # Step 3: Create the mapping
    icallbbtocallee = {}
    # For each key in icallinstocallee.json
    for key, callees in icallinstocallee.items():
        # Find the same key in callinstobb and get its value (the bb)
        if key in callinstobb:
            bb = callinstobb[key]
            new_callees = []
            for callee in callees:
                if callee in instofunc:
                    new_callees.append(instofunc[callee])
                    # print(f"callee:{callee} -> {instofunc[callee]} ")
                else:
                    new_callees.append(callee)
            assert len(callees)  == len(new_callees)
            icallbbtocallee[bb] = new_callees
            # if callee in instofunc:
            #     funcaddr = instofunc[callee]
            #     icallbbtocallee[bb] = funcaddr
            #     print(f"callee:{callee}, funcadd:{funcaddr}\n")
            # else:
            #     icallbbtocallee[bb]=callee

            
    # Step 4: Save the mapping to the output file
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(icallbbtocallee, f, indent=4)
    
    print(f"Mapping created and saved to {output_path}")

    if icallbbtocallee:
        return icallbbtocallee.keys()
    return []

def main():
    # Get folder paths from user
    if len(sys.argv) != 5:
        sys.exit(1)
        
    binary_name = sys.argv[1]
    angrcfginfo = sys.argv[2]  # Folder containing the jmptable.json
    sourceinfo = sys.argv[3]
    output_folder = sys.argv[4]
    
    # Process the files
    process_icall(binary_name, angrcfginfo, sourceinfo, output_folder)

if __name__ == "__main__":
    main()
