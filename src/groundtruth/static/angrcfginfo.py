"""Static graph-construction pipeline driven by angr and PalmTree embeddings."""

from collections import defaultdict
import angr
import logging
from typing import Optional, List, Tuple, Dict, Set
import argparse
import sys
import json
import os
import glob
import dgl
import hashlib
from logger import Logger
import eval_utils as utils

import torch as th
import numpy as np
import shutil
import gzip
from processjumptable import correct_jump_mappings 
from processtailcall import process_itc_files
from processicall import process_icall


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", "..", ".."))
PALMTREE_DIR = os.path.join(REPO_ROOT, "src", "groundtruth", "static", "palmtree")
LOG_ROOT = os.environ.get("DUALICFLOW_LOG_DIR", os.path.join(REPO_ROOT, "logs"))

logger = Logger(base_dir=LOG_ROOT)

NINST_ADDRS = 70  # Number of instruction addresses to use (vectors)


palmtree = utils.UsableTransformer(
    model_path=os.environ.get("PALMTREE_MODEL", os.path.join(PALMTREE_DIR, "transformer.ep19")),
    vocab_path=os.environ.get("PALMTREE_VOCAB", os.path.join(PALMTREE_DIR, "vocab")),
)



class GlobalState:
    def __init__(self):
        self.code_node_id = 0
        self.data_node_id = 0
        self.func_node_id = 0
        self.addr_min = 0xffffffff
        self.addr_max = 0
        self.graph_list = []
        self.asm_dict = {}
        self.naming = 0
        
        
class CodeTrainNode:
    """Represents a code block node in the graph."""

    def __init__(self, node, state, is_func_node=False, encode_embeddings=True):
        """Initialize a code node from an angr node."""
        self.id = state.code_node_id
        state.code_node_id += 1
        self.addr = node.addr
        self.func_addr = node.function_address
        self.func_node = is_func_node
        if self.func_addr == node.function_address:
            self.func_node = True          
        self.text = []
        self.lastinstopcode = ""

        # Extract and process instructions
        insns = node.block.capstone.insns
        for insn in insns:
            tmp = insn.mnemonic + ' ' + insn.op_str
            self.text.append(tmp.replace(',', '').replace('[', '[ ').replace(']', ' ]'))

        if len(self.text) == 0:
            self.text = ['nop']

        # Fix for getting the last instruction opcode
        if len(self.text) > 0:
            self.lastinstopcode = self.text[-1].split()[0]

        state.asm_dict[self.addr] = self.text

        # Generate embeddings immediately by default. Large binaries use the
        # batched graph-builder path below to avoid one model call per block.
        self.embeddings = None
        self.avg = None
        if encode_embeddings:
            self.embeddings = palmtree.encode(self.text)
            self.avg = self.embeddings.mean(axis=0)
            self.embeddings = self.embeddings[:NINST_ADDRS]

        # Update global min/max address
        state.addr_max = max(state.addr_max, self.addr)
        state.addr_min = min(state.addr_min, self.addr)


class DataTrainNode:
    """Represents a data node in the graph."""

    def __init__(self, addr, state, extern_func_name=None):
        """Initialize a data node."""
        self.id = state.data_node_id
        state.data_node_id += 1
        self.addr = addr
        self.extern_func_name = extern_func_name

        # Update global min/max address
        state.addr_max = max(state.addr_max, self.addr)
        state.addr_min = min(state.addr_min, self.addr)


def is_call_instruction(insn):
    """Check if an instruction is any call instruction."""
    return insn.insn.mnemonic.startswith('call')

def is_jump_instruction(insn):
    """Check if an instruction is any jump instruction."""
    return insn.insn.mnemonic.startswith('j')

# Method using Angr's cle loader
def is_pie_angr(binary_path: str) -> bool:
    """
    Check if the binary at `binary_path` is a PIE (Position Independent Executable).
    
    Args:
        binary_path (str): Path to the binary.

    Returns:
        bool: True if PIE, False otherwise.
    """
    project = angr.Project(binary_path, auto_load_libs=False)
    return project.loader.main_object.pic

class AngrAnalyzer:
    def __init__(self, binary_path: str):
        """Initialize the analyzer with a binary path."""
        logging.getLogger('angr').setLevel(logging.WARNING)
        try:
            if is_pie_angr(binary_path):
                self.proj = angr.Project(binary_path,
                                   auto_load_libs=False,main_opts={'base_addr': 0x0})
            else:
                self.proj = angr.Project(binary_path,
                                   auto_load_libs=False)
            
            print(f"angr default base: {hex(self.proj.loader.main_object.mapped_base)}")
        except Exception as e:
            print(f"Error loading binary {binary_path}: {str(e)}")
            sys.exit(1)
        print("Generating CFG...")
        try:
            self.cfg = self.proj.analyses.CFGFast(
            cross_references=True,
            resolve_indirect_jumps=False,
            )
            print("CFG generation complete!")
        except Exception as e:
            print(f"Error generating CFG: {str(e)}")
            sys.exit(1)
        self._cfg_nodes_cache = list(self.cfg.model.nodes())
        self._node_by_addr_cache = {}
        for node in self._cfg_nodes_cache:
            if hasattr(node, "addr"):
                self._node_by_addr_cache.setdefault(node.addr, node)
        self._containing_block_cache = {}
        self._function_addrs_cache = None
        self._bb_to_func_cache = None
        self._instr_to_func_cache = None
        self._node_addrs_cache = None
        self._call_to_bb_cache = None
        self._jump_to_bb_cache = None
        self._direct_call_to_func_cache = None
        self._jump_to_func_cache = None
        self._call_to_next_cache = None
        self._next_block_after_call_cache = {}

    def get_function_hashes(self, sourcelist,bbtofunc):
        res =  {}
        for source in sourcelist:
            if source in bbtofunc:
                funcaddr = bbtofunc[source]
                function_hash = self.get_function_asm_hash_from_cfg(self.cfg, funcaddr)
                res[source] = function_hash
        return res        
            
        
    def get_function_asm_hash_from_cfg(self, cfg: angr.analyses.cfg.CFGBase, func_addr_hex: str) -> str:
        """
        Compute a SHA-256 hash of the assembly instructions for a function in the CFG.

        Args:
            cfg: An angr CFG (e.g., CFGFast) object.
            func_addr_hex: Hexadecimal string of the function address (e.g., '0x4005f6').

        Returns:
            A SHA-256 hexadecimal hash string representing the function's assembly.
        """
        try:
            func_addr = int(func_addr_hex, 16)
        except ValueError:
            raise ValueError(f"Invalid hexadecimal address: {func_addr_hex}")

        func = cfg.kb.functions.get(func_addr)
        if func is None:
            raise ValueError(f"Function at {func_addr_hex} not found in CFG.")

        disasm_lines = []
        for block in func.blocks:
            for insn in block.capstone.insns:
                insn_str = f"{insn.mnemonic} {insn.op_str}"
                disasm_lines.append(insn_str)
                
        asm_str = "\n".join(disasm_lines)
        return hashlib.sha256(asm_str.encode('utf-8')).hexdigest()
       

    def save_graph(self, binary_name, jsonfolder, graphdir, graph, nodelookup, binary_index):
        """Save the generated graph and related data to files."""

        # Ensure the graph directory exists
        os.makedirs(graphdir, exist_ok=True)

        # Save the graph to a temporary file
        temp_graph_path = os.path.join(graphdir, f"{binary_index}.graph")
        dgl.data.utils.save_graphs(temp_graph_path, [graph])

        # Compress the graph file using gzip
        compressed_graph_path = temp_graph_path + '.gz'
        with open(temp_graph_path, 'rb') as f_in, gzip.open(compressed_graph_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        # Remove the temporary uncompressed graph file
        os.remove(temp_graph_path)

        # Save the node lookup dictionary as JSON
        os.makedirs(jsonfolder, exist_ok=True)
        nodelookup_path = os.path.join(jsonfolder, f"{binary_name}_nodelookup.json")
        with open(nodelookup_path, 'w') as f:
            json.dump({
                hex(addr.item() if hasattr(addr, "item") else addr): node.id
                for addr, node in nodelookup.items()
                if node is not None and addr is not None
            }, f, indent=4)

        print(f"Compressed graph saved to: {compressed_graph_path}")
        print(f"Node lookup saved to: {nodelookup_path}")

    def _encode_code_node_batch(self, batch):
        """Encode a group of code nodes with one PalmTree call."""
        flat_text = []
        spans = []
        for node in batch:
            start = len(flat_text)
            flat_text.extend(node.text)
            spans.append((node, start, len(flat_text)))

        if not flat_text:
            return

        encoded = palmtree.encode(flat_text)
        for node, start, end in spans:
            node_embeddings = encoded[start:end]
            node.avg = node_embeddings.mean(axis=0)
            node.embeddings = node_embeddings[:NINST_ADDRS]

    def encode_code_nodes(self, code_nodes):
        """Encode code-node instruction embeddings in batches."""
        if not code_nodes:
            return

        try:
            batch_size = int(os.environ.get("PALMTREE_BATCH_SIZE", "512"))
        except ValueError:
            batch_size = 512
        batch_size = max(1, batch_size)

        batch = []
        batch_inst_count = 0
        encoded_nodes = 0
        next_report = 10000
        total_nodes = len(code_nodes)

        for node in code_nodes:
            node_inst_count = len(node.text)
            if batch and batch_inst_count + node_inst_count > batch_size:
                self._encode_code_node_batch(batch)
                encoded_nodes += len(batch)
                if encoded_nodes >= next_report:
                    print(f"Encoded PalmTree code nodes {encoded_nodes}/{total_nodes}", flush=True)
                    next_report += 10000
                batch = []
                batch_inst_count = 0

            batch.append(node)
            batch_inst_count += node_inst_count

        if batch:
            self._encode_code_node_batch(batch)
            encoded_nodes += len(batch)

        print(f"Encoded PalmTree code nodes {encoded_nodes}/{total_nodes}", flush=True)

    def build_graph(self,binary_file, state):
        """Build a graph representation of the binary."""
        logger.info(f"Processing: {binary_file}")

        logger.info(f"Generated CFG for {binary_file}")

        # Reset node counters and address ranges
        state.code_node_id = 0
        state.data_node_id = 0
        state.func_node_id = 0
        state.addr_min = 0xffffffff
        state.addr_max = 0

        # Initialize data structures
        code_nodes = []
        data_nodes = []
        code2code_edges = []
        codexrefcode_edges = []
        codecall_edges = []
        codejump_edges = []
        codexrefdata_edges = []
        dataxrefcode_edges = []
        dataxrefdata_edges = []
        jumpopcode = ['jmp', 'jz','jnz','jnbe','jnb','jnae','jna','jnle','jnl','jnge','jng']

        # Build lookup tables
        node_lookup = defaultdict(lambda: None)
        # functobb = defaultdict(list)  # Changed to use list as default value
        # sym_lookup = defaultdict(lambda: None)
        # func_lookup = defaultdict(lambda: None)

        
        sym_lookup = defaultdict(lambda: None)
        func_lookup = defaultdict(lambda: None)

            # name -> addr
            # addr -> name
        for faddr, func in self.proj.kb.functions.items():
            func_lookup[func.name] = faddr
            sym_lookup[faddr] = func.name

        # free is a hook function of angr
        # change the virtual address to rebased_addr
        funcfree = "free"
        a = self.proj.loader.find_symbol(funcfree)
        if a is not None:
            faddr = a.rebased_addr
            func_lookup[func] = faddr
            sym_lookup[faddr] = func


        logger.info("Processed function symbols")

        # Step 1: Create code nodes
        for node in self.cfg.graph.nodes():
            if node.block is not None:
                new_node = CodeTrainNode(node, state, encode_embeddings=False)
                new_node.func_addr = node.function_address
                code_nodes.append(new_node)
                node_lookup[node.addr] = new_node

        self.encode_code_nodes(code_nodes)

        logger.info(f"Created {len(code_nodes)} code nodes")

        # Step 2: Create code edges
        for edge in self.cfg.graph.edges:
            node0 = node_lookup[edge[0].addr]
            node1 = node_lookup[edge[1].addr]
            if node0 is not None and node1 is not None:
                new_edge = (node0.id, node1.id)
                insns = edge[0].block.capstone.insns
                if len(insns) == 0:
                    continue
                if insns[-1].mnemonic in jumpopcode:
                    codejump_edges.append(new_edge)
                elif insns[-1].mnemonic == 'call':
                    codecall_edges.append(new_edge)
                else:
                    code2code_edges.append(new_edge)

        logger.info(f"Created code edges: {len(code2code_edges)} code2code, {len(codecall_edges)} call, {len(codejump_edges)} jump")

        # Step 3: Create cross-reference edges
        for dst in self.proj.kb.xrefs.xrefs_by_dst:
            xrefs = self.proj.kb.xrefs.xrefs_by_dst[dst]
            for xref in xrefs:
                node0_addr = xref.block_addr if xref.block_addr is not None else xref.ins_addr
                node0 = node_lookup[node0_addr]
                node1 = node_lookup[dst]

                # Create missing nodes
                if node0 is None:
                    new_node = DataTrainNode(node0_addr, state, sym_lookup[node0_addr])
                    data_nodes.append(new_node)
                    node_lookup[node0_addr] = new_node
                    node0 = new_node

                if node1 is None:
                    new_node = DataTrainNode(dst, state, sym_lookup[dst])
                    data_nodes.append(new_node)
                    node_lookup[dst] = new_node
                    node1 = new_node

                # Add appropriate edge type
                # new_edge = (node0.id, node1.id)
                # if isinstance(node0, CodeTrainNode) and isinstance(node1, CodeTrainNode):
                #     codexrefcode_edges.append(new_edge)
                # elif isinstance(node0, CodeTrainNode) and isinstance(node1, DataTrainNode):
                #     codexrefdata_edges.append(new_edge)
                # elif isinstance(node0, DataTrainNode) and isinstance(node1, CodeTrainNode):
                #     dataxrefcode_edges.append(new_edge)
                # elif isinstance(node0, DataTrainNode) and isinstance(node1, DataTrainNode):
                #     dataxrefdata_edges.append(new_edge)
                # Track seen edges to avoid duplicates
                seen_codexrefdata_edges = set()

                # Add appropriate edge type
                new_edge = (node0.id, node1.id)
                if isinstance(node0, CodeTrainNode) and isinstance(node1, CodeTrainNode):
                    codexrefcode_edges.append(new_edge)
                elif isinstance(node0, CodeTrainNode) and isinstance(node1, DataTrainNode):
                    if new_edge not in seen_codexrefdata_edges:
                        codexrefdata_edges.append(new_edge)
                        seen_codexrefdata_edges.add(new_edge)
                elif isinstance(node0, DataTrainNode) and isinstance(node1, CodeTrainNode):
                    dataxrefcode_edges.append(new_edge)
                elif isinstance(node0, DataTrainNode) and isinstance(node1, DataTrainNode):
                    dataxrefdata_edges.append(new_edge)
                            
        logger.info(f"Created cross-reference edges")


        # Step 3: Create code2funchead edges instead of func node
        code2funchead_edges = []
        func_entry_map = {}  # func_addr -> code node id

        # Find function entry blocks and assign
        for node in code_nodes:
            if node.func_node:
                key = node.func_addr.item() if hasattr(node.func_addr, "item") else node.func_addr
                func_entry_map[key] = node.id

        for node in code_nodes:
            key = node.func_addr.item() if hasattr(node.func_addr, "item") else node.func_addr
            head_id = func_entry_map.get(key)
            if head_id is not None and head_id != node.id:
                code2funchead_edges.append((node.id, head_id))

        # Replace original code2func_edges with code2funchead in graph_data
        graph_data = {
            ('code', 'code2code_edges', 'code'): code2code_edges,
            ('code', 'codecall_edges', 'code'): codecall_edges,
            ('code', 'codejump_edges', 'code'): codejump_edges,
            ('code', 'code2funchead', 'code'): code2funchead_edges,
            ('code', 'codexrefcode_edges', 'code'): codexrefcode_edges,
            ('code', 'codexrefdata_edges', 'data'): codexrefdata_edges,
            ('data', 'dataxrefcode_edges', 'code'): dataxrefcode_edges,
            ('data', 'dataxrefdata_edges', 'data'): dataxrefdata_edges
        }
        num_nodes_dict = {
            'code': len(code_nodes),
            'data': len(data_nodes),
        }

        # Create heterogeneous graph
        g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        logger.info("Created heterogeneous graph")

        # Normalize addresses between 0 and 1
        new_min, new_max = 0, 1
        addr_range = state.addr_max - state.addr_min

        # Process code node features
        for node in code_nodes:
            # Normalize addresses
            node.addr = th.tensor((node.addr - state.addr_min) / addr_range * (new_max - new_min) + new_min)
            node.func_addr = th.tensor((node.func_addr - state.addr_min) / addr_range * (new_max - new_min) + new_min)

            # Pad embeddings to fixed length
            padded_embeddings = np.array(list(node.embeddings) + [[0]*128]*(NINST_ADDRS - len(node.embeddings)))
            node.embeddings = th.tensor(padded_embeddings).view(-1)

            # Concatenate with address information
            node.embeddings = th.cat((node.addr.view(1), node.func_addr.view(1), node.embeddings)).float()

            # Process average embeddings
            node.avg = th.from_numpy(node.avg)
            node.avg = th.cat((node.addr.view(1), node.func_addr.view(1), node.avg)).float()

        # Process data node features
        for node in data_nodes:
            node.addr = th.tensor((node.addr - state.addr_min) / addr_range * (new_max - new_min) + new_min)

        # Assign features to graph nodes
        if len(code_nodes) > 0:
            g.nodes['code'].data['feat'] = th.stack([node.embeddings for node in code_nodes])
            g.nodes['code'].data['featmean'] = th.stack([node.avg for node in code_nodes])

        if len(data_nodes) > 0:
            g.nodes['data'].data['feat'] = th.stack([node.addr.view(1) for node in data_nodes])

        # logger.info(f"Node statistics:")
        # logger.info(f"  Code nodes: {len(code_nodes)}")
        # logger.info(f"  Data nodes: {len(data_nodes)}")
        # logger.info(f"Edge statistics:")
        # logger.info(f"  code2code edges: {len(code2code_edges)}")
        # logger.info(f"  codecall edges: {len(codecall_edges)}")
        # logger.info(f"  codejump edges: {len(codejump_edges)}")
        # logger.info(f" code2funchead edges: {len(code2funchead_edges)}")
        # logger.info(f"  codexrefcode edges: {len(codexrefcode_edges)}")
        # logger.info(f"  codexrefdata edges: {len(codexrefdata_edges)}")
        # logger.info(f"  dataxrefcode edges: {len(dataxrefcode_edges)}")
        # logger.info(f"  dataxrefdata edges: {len(dataxrefdata_edges)}")

        return g, node_lookup

    def get_containing_block(self, address: int) -> Optional[angr.knowledge_plugins.cfg.cfg_node.CFGNode]:
        """Find the basic block containing the given address."""
        try:
            cached = self._containing_block_cache.get(address)
            if address in self._containing_block_cache:
                return cached

            node = self._node_by_addr_cache.get(address)
            if node is not None:
                self._containing_block_cache[address] = node
                return node

            for node in self._cfg_nodes_cache:
                if node.addr <= address < node.addr + node.size:
                    self._containing_block_cache[address] = node
                    return node
            self._containing_block_cache[address] = None
            return None
        except Exception as e:
            print(f"Error finding block for address 0x{address:x}: {str(e)}")
            return None

    def get_next_block_after_call(self, block_node):
        """Get the address of the block that follows a call"""
        cache_key = getattr(block_node, "addr", None)
        if cache_key in self._next_block_after_call_cache:
            return self._next_block_after_call_cache[cache_key]
        try:
            successors = list(self.cfg.graph.successors(block_node))
            if not successors:
                self._next_block_after_call_cache[cache_key] = None
                return None
            block = block_node.block
            if not block:
                self._next_block_after_call_cache[cache_key] = None
                return None
            instructions = list(block.capstone.insns)
            if not instructions:
                self._next_block_after_call_cache[cache_key] = None
                return None
            last_insn = instructions[-1]
            # 1. First try using instruction layout
            call_next_addr = last_insn.address + last_insn.size
            # 2. Check for exact address match
            for succ in successors:
                if succ.addr == call_next_addr:
                    self._next_block_after_call_cache[cache_key] = succ.addr
                    return succ.addr
        except Exception as e:
            print(f"Error in get_next_block_after_call: {e}")
        self._next_block_after_call_cache[cache_key] = None
        return None

    def get_all_function_addresses(self) -> List[int]:
        """Get all function addresses in the binary."""
        if self._function_addrs_cache is not None:
            return list(self._function_addrs_cache)
        res = []
        try:
            for func_addr, func in self.cfg.functions.items():
                if func.is_simprocedure or func.is_plt:
                    continue
                if not func.blocks:  # Check if blocks list is empty
                    continue
                res.append(func_addr)
            self._function_addrs_cache = list(res)
            return res 
        except Exception as e:
            print(f"Error getting function addresses: {str(e)}")
            return []

    def get_function_blocks(self, func_addr: int) -> List[int]:
        """Get all basic block addresses for a function."""
        try:
            func = self.cfg.functions.get(func_addr)
            if func:
                return list(func.block_addrs)
            return []
        except Exception as e:
            print(f"Error getting blocks for function 0x{func_addr:x}: {str(e)}")
            return []

    def get_all_blocks_to_functions(self) -> Dict[int, int]:
        """Create mapping of basic blocks to their containing functions."""
        if self._bb_to_func_cache is not None:
            return dict(self._bb_to_func_cache)
        bb_to_func = {}
        try:
            for func_addr, func in self.cfg.functions.items():
                for block_addr in func.block_addrs:
                    bb_to_func[block_addr] = func_addr
            self._bb_to_func_cache = dict(bb_to_func)
            return bb_to_func
        except Exception as e:
            print(f"Error creating block to function mapping: {str(e)}")
            return {}
        
    def get_all_instructions_to_functions(self) -> Dict[int, int]:
        """Create mapping of all instructions to their containing functions."""
        if self._instr_to_func_cache is not None:
            return dict(self._instr_to_func_cache)
        all_instr_to_func = {}
    
        # Create a mapping of block addresses to graph nodes for efficiency
        addr_to_node = self._node_by_addr_cache
    
        try:
            for func_addr, func in self.cfg.functions.items():
                if not func.block_addrs:
                    continue
            
                # Iterate through all basic blocks in this function
                for block_addr in func.block_addrs:
                    if block_addr in addr_to_node:
                        node = addr_to_node[block_addr]
                        if hasattr(node, 'instruction_addrs'):
                            # Map all instructions in this block to the function
                            for instr_addr in node.instruction_addrs:
                                all_instr_to_func[instr_addr] = func_addr
                    else:
                        print(f"Warning: Block {hex(block_addr)} not found in CFG graph for function {hex(func_addr)}")
                    
            self._instr_to_func_cache = dict(all_instr_to_func)
            return all_instr_to_func
    
        except Exception as e:
            print(f"Error creating instruction to function mapping: {str(e)}")
            return {}

    def get_all_node_addresses(self) -> List[int]:
        """Get all basic block (node) addresses in the binary in order."""
        if self._node_addrs_cache is not None:
            return list(self._node_addrs_cache)
        try:
            # Sort nodes by address to maintain order
            nodes = sorted(self._cfg_nodes_cache, key=lambda node: node.addr)
            self._node_addrs_cache = [node.addr for node in nodes if hasattr(node, 'addr')]
            return list(self._node_addrs_cache)
        except Exception as e:
            print(f"Error getting node addresses: {str(e)}")
            return []

    def find_all_call_instructions(self) -> Dict[int, int]:
        """Find all call instructions and map them to their containing basic blocks."""
        if self._call_to_bb_cache is not None:
            return dict(self._call_to_bb_cache)
        call_to_bb = {}
        try:
            for node in self._cfg_nodes_cache:
                if not node.block:
                    continue
                for insn in node.block.capstone.insns:
                    if is_call_instruction(insn):
                        call_to_bb[insn.address] = node.addr
            self._call_to_bb_cache = dict(call_to_bb)
            return call_to_bb
        except Exception as e:
            print(f"Error finding call instructions: {str(e)}")
            return {}

    def find_all_jump_instructions(self) -> Dict[int, int]:
        """Find all jump instructions and map them to their containing basic blocks."""
        if self._jump_to_bb_cache is not None:
            return dict(self._jump_to_bb_cache)
        jump_to_bb = {}
        try:
            for node in self._cfg_nodes_cache:
                if not node.block:
                    continue
                for insn in node.block.capstone.insns:
                    if is_jump_instruction(insn):
                        jump_to_bb[insn.address] = node.addr
            self._jump_to_bb_cache = dict(jump_to_bb)
            return jump_to_bb
        except Exception as e:
            print(f"Error finding jump instructions: {str(e)}")
            return {}
            
    def find_jump_instructions_to_functions(self) -> Dict[int, int]:
        """Find all jump instructions and map them to their containing functions."""
        if self._jump_to_func_cache is not None:
            return dict(self._jump_to_func_cache)
        jump_to_func = {}
        try:
            # Get jump instructions to basic blocks mapping
            jump_to_bb = self.find_all_jump_instructions()
            
            # Get basic blocks to functions mapping
            bb_to_func = self.get_all_blocks_to_functions()
            
            # Create direct mapping from jump instructions to functions
            for jump_addr, bb_addr in jump_to_bb.items():
                if bb_addr in bb_to_func:
                    jump_to_func[jump_addr] = bb_to_func[bb_addr]
                    
            self._jump_to_func_cache = dict(jump_to_func)
            return jump_to_func
        except Exception as e:
            print(f"Error finding jump instructions to functions mapping: {str(e)}")
            return {}

    def analyze_all_calls_to_next_block(self) -> Dict[int, Optional[int]]:
        """Analyze all calls and map them to their next basic blocks."""
        if self._call_to_next_cache is not None:
            return dict(self._call_to_next_cache)
        results = {}
        try:
            call_to_bb = self.find_all_call_instructions()
            total = len(call_to_bb)
            print(f"Found {total} call instructions")

            unique_bb_addrs = list(dict.fromkeys(call_to_bb.values()))
            bb_to_next = {}
            for i, bb_addr in enumerate(unique_bb_addrs, 1):
                if i % 100 == 0:
                    print(f"Processing call basic block {i}/{len(unique_bb_addrs)}...")

                block_node = self.get_containing_block(bb_addr)
                if block_node is None:
                    continue

                next_block_addr = self.get_next_block_after_call(block_node)
                if next_block_addr is not None:
                    bb_to_next[bb_addr] = next_block_addr

            for call_addr, bb_addr in call_to_bb.items():
                next_block_addr = bb_to_next.get(bb_addr)
                if next_block_addr is not None:
                    results[call_addr] = next_block_addr
                    
            self._call_to_next_cache = dict(results)
            return results
        except Exception as e:
            print(f"Error analyzing calls to next blocks: {str(e)}")
            return {}

    def find_direct_call_instructions(self) -> Dict[int, int]:
        """Find all direct call instructions and map them to their target function addresses."""
        if self._direct_call_to_func_cache is not None:
            return dict(self._direct_call_to_func_cache)
        direct_call_to_func = {}
        try:
            for node in self._cfg_nodes_cache:
                if not node.block:
                    continue
                for insn in node.block.capstone.insns:
                    if is_call_instruction(insn):
                        call_addr = insn.address
                        if insn.operands and hasattr(insn.operands[0], 'imm'):
                            callee_addr = insn.operands[0].imm
                            callee_func = self.cfg.functions.get(callee_addr)
                            if callee_func:
                                direct_call_to_func[call_addr] = callee_addr
            self._direct_call_to_func_cache = dict(direct_call_to_func)
            return direct_call_to_func
        except Exception as e:
            print(f"Error finding direct call instructions: {str(e)}")
            return {}

    def is_return_instruction(self, insn):
        """Check if an instruction is a return instruction."""
        return insn.insn.mnemonic.startswith('ret')

    def get_function_return_blocks(self, func_addr: int) -> List[int]:
        """Get all basic block addresses containing return instructions for a function."""
        ret_blocks = []
        try:
            func = self.cfg.functions.get(func_addr)
            if not func:
                return []
                
            for block_addr in func.block_addrs:
                node = self.get_containing_block(block_addr)
                if not node or not node.block:
                    continue
                    
                insns = list(node.block.capstone.insns)
                if not insns:
                    continue
                    
                # Check if last instruction is a return
                last_insn = insns[-1]
                if self.is_return_instruction(last_insn):
                    ret_blocks.append(block_addr)
                    
            return ret_blocks
        except Exception as e:
            print(f"Error getting return blocks for function 0x{func_addr:x}: {str(e)}")
            return []
            
    def get_callee_to_call_instructions(self) -> Dict[int, List[int]]:
        """Get a mapping of function addresses to the call instructions that call them."""
        callee_to_call_ins = {}
        try:
            # Get the direct call to function mapping
            direct_call_to_func = self.find_direct_call_instructions()
            
            # Invert the mapping
            for call_addr, func_addr in direct_call_to_func.items():
                if func_addr not in callee_to_call_ins:
                    callee_to_call_ins[func_addr] = []
                callee_to_call_ins[func_addr].append(call_addr)
                
            return callee_to_call_ins
        except Exception as e:
            print(f"Error getting callee to call instructions mapping: {str(e)}")
            return {}

    def is_tail_call_instruction(self, insn):
        """Check if an instruction is a direct tail call instruction (jmp to a function)."""
        # This is typically a jmp instruction to a function address
        return insn.insn.mnemonic.startswith('jmp')

    def load_direct_tail_calls_from_file(self, dtc_file_path: str) -> List[int]:
        """Load direct tail call instruction addresses from a JSON file."""
        try:
            with open(dtc_file_path, 'r') as f:
                data = json.load(f)
            
            # Extract direct tail call addresses and convert from hex strings to integers
            dtc_addresses = [int(addr, 16) for addr in data.get("dtailcall_addresses", [])]
            return dtc_addresses
        except Exception as e:
            print(f"Error loading direct tail call addresses from {dtc_file_path}: {str(e)}")
            return []

    def find_direct_tail_call_targets(self, dtc_addresses: List[int]) -> Dict[int, int]:
        """Find the target function for each direct tail call instruction."""
        dtc_to_callee = {}
        try:
            for dtc_addr in dtc_addresses:
                # Get the basic block containing this instruction
                block_node = self.get_containing_block(dtc_addr)
                if not block_node or not block_node.block:
                    continue
                    
                # Find the tail call instruction
                for insn in block_node.block.capstone.insns:
                    if insn.address == dtc_addr and self.is_tail_call_instruction(insn):
                        # Check if it's a direct jump with immediate operand
                        if insn.operands and hasattr(insn.operands[0], 'imm'):
                            callee_addr = insn.operands[0].imm
                            callee_func = self.cfg.functions.get(callee_addr)
                            if callee_func:
                                dtc_to_callee[dtc_addr] = callee_addr
                        break
                    
            return dtc_to_callee
        except Exception as e:
            print(f"Error finding direct tail call targets: {str(e)}")
            return {}


    def save_direct_tail_call_to_function(self, dtc_file_path: str, output_file: str):
        """Save direct tail call instruction to target function mapping to a JSON file."""
        try:
            # Load direct tail call addresses from the input file
            dtc_addresses = self.load_direct_tail_calls_from_file(dtc_file_path)
            output_data = {}
            dtc_to_callee = {}

            if not dtc_addresses:
                print(f"No direct tail call addresses found in {dtc_file_path}")
            else:
                print(f"Loaded {len(dtc_addresses)} direct tail call addresses")
                
                # Find the target function for each direct tail call
                dtc_to_callee = self.find_direct_tail_call_targets(dtc_addresses)

                # Convert to hex string format for output
                output_data = {
                    f"0x{dtc_addr:x}": f"0x{callee_addr:x}" 
                    for dtc_addr, callee_addr in dtc_to_callee.items()
                }

            # Always write output, even if it's an empty dictionary
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"Direct tail call to callee function mapping saved to {output_file}")
            print(f"Mapped {len(dtc_to_callee)} out of {len(dtc_addresses)} direct tail calls")

        except Exception as e:
            print(f"Error saving direct tail call to function mapping: {str(e)}")

    def save_callee_to_call_instructions(self, output_file: str):
        """Save function (callee) to call instruction addresses mapping to a JSON file."""
        callee_to_call_ins = self.get_callee_to_call_instructions()
        output_data = {f"0x{func_addr:x}": [f"0x{call_addr:x}" for call_addr in call_addrs] 
                      for func_addr, call_addrs in callee_to_call_ins.items()}
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Function (callee) to call instruction mapping saved to {output_file}")
        except Exception as e:
            print(f"Error saving function to call instructions mapping: {str(e)}")

    def save_all_function_addresses(self, output_file: str):
        """Save all function addresses to a JSON file."""
        func_addrs = self.get_all_function_addresses()
        output_data = {"funcaddr": [f"0x{addr:x}" for addr in func_addrs]}
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Function addresses saved to {output_file}")
        except Exception as e:
            print(f"Error saving function addresses: {str(e)}")

    def save_node_addresses(self, output_file: str):
        """Save all basic block (node) addresses to a JSON file."""
        node_addrs = self.get_all_node_addresses()
        output_data = {"nodeaddres": [f"0x{addr:x}" for addr in node_addrs]}
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Node addresses saved to {output_file}")
        except Exception as e:
            print(f"Error saving node addresses: {str(e)}")

    def save_function_to_blocks(self, output_file: str):
        """Save function to blocks mapping to a JSON file."""
        func_to_blocks = {}
        func_addrs = self.get_all_function_addresses()
        total = len(func_addrs)
        for i, func_addr in enumerate(func_addrs, 1):
            if i % 100 == 0:
                print(f"Processing function to blocks mapping {i}/{total}...")
            func_addr_str = f"0x{func_addr:x}"
            block_addrs = self.get_function_blocks(func_addr)
            func_to_blocks[func_addr_str] = [f"0x{addr:x}" for addr in block_addrs]
        try:
            with open(output_file, 'w') as f:
                json.dump(func_to_blocks, f, indent=2)
            print(f"Function to blocks mapping saved to {output_file}")
        except Exception as e:
            print(f"Error saving function to blocks mapping: {str(e)}")

    def save_block_to_function(self, output_file: str):
        """Save block to function mapping to a JSON file."""
        bb_to_func = self.get_all_blocks_to_functions()
        output_data = {f"0x{bb_addr:x}": f"0x{func_addr:x}" for bb_addr, func_addr in bb_to_func.items()}
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Block to function mapping saved to {output_file}")
            return output_data
        except Exception as e:
            print(f"Error saving block to function mapping: {str(e)}")
        return output_data
    
    def save_first_block_instructions_to_function(self, output_file: str):
        """Save first block instruction to function mapping to a JSON file."""
        first_bb_instr_to_func = self.get_all_instructions_to_functions()
        output_data = {f"0x{instr_addr:x}": f"0x{func_addr:x}" for instr_addr, func_addr in first_bb_instr_to_func.items()}
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"First block instruction to function mapping saved to {output_file}")
            return output_data
        except Exception as e:
            print(f"Error saving first block instruction to function mapping: {str(e)}")
        return output_data
            

    def save_call_to_block(self, output_file: str):
        """Save call instruction to containing block mapping to a JSON file."""
        call_to_bb = self.find_all_call_instructions()
        output_data = {f"0x{call_addr:x}": f"0x{bb_addr:x}" for call_addr, bb_addr in call_to_bb.items()}
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Call instruction to block mapping saved to {output_file}")
        except Exception as e:
            print(f"Error saving call instruction to block mapping: {str(e)}")

    def save_jump_to_block(self, output_file: str):
        """Save jump instruction to containing block mapping to a JSON file."""
        jump_to_bb = self.find_all_jump_instructions()
        output_data = {f"0x{jump_addr:x}": f"0x{bb_addr:x}" for jump_addr, bb_addr in jump_to_bb.items()}
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Jump instruction to block mapping saved to {output_file}")
        except Exception as e:
            print(f"Error saving jump instruction to block mapping: {str(e)}")
            
    def save_jump_to_function(self, output_file: str):
        """Save jump instruction to containing function mapping to a JSON file."""
        jump_to_func = self.find_jump_instructions_to_functions()
        output_data = {f"0x{jump_addr:x}": f"0x{func_addr:x}" for jump_addr, func_addr in jump_to_func.items()}
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Jump instruction to function mapping saved to {output_file}")
        except Exception as e:
            print(f"Error saving jump instruction to function mapping: {str(e)}")

    def save_call_blocks_list(self, output_file: str):
        """Save a list of all basic blocks containing call instructions to a JSON file."""
        call_to_bb = self.find_all_call_instructions()
        # Extract unique basic block addresses that contain call instructions
        unique_call_bbs = list(set(call_to_bb.values()))
        # Convert to hex strings
        hex_call_bbs = [f"0x{bb_addr:x}" for bb_addr in unique_call_bbs]
        # Create the output data structure with "callbb" as the key
        output_data = {"callbb": hex_call_bbs}
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Call blocks list saved to {output_file}")
        except Exception as e:
            print(f"Error saving call blocks list: {str(e)}")

    def save_jump_blocks_list(self, output_file: str):
        """Save a list of all basic blocks containing jump instructions to a JSON file."""
        jump_to_bb = self.find_all_jump_instructions()
        # Extract unique basic block addresses that contain jump instructions
        unique_jump_bbs = list(set(jump_to_bb.values()))
        # Convert to hex strings
        hex_jump_bbs = [f"0x{bb_addr:x}" for bb_addr in unique_jump_bbs]
        # Create the output data structure with "jmpbb" as the key
        output_data = {"jumpbb": hex_jump_bbs}
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Jump blocks list saved to {output_file}")
        except Exception as e:
            print(f"Error saving jump blocks list: {str(e)}")

    def save_call_to_after_call(self, output_file: str):
        """Save call instruction to next block mapping to a JSON file."""
        call_to_next = self.analyze_all_calls_to_next_block()
        output_data = {f"0x{call_addr:x}": f"0x{next_addr:x}" if next_addr is not None else None 
                      for call_addr, next_addr in call_to_next.items()}
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Call instruction to next block mapping saved to {output_file}")
        except Exception as e:
            print(f"Error saving call instruction to next block mapping: {str(e)}")

    def save_direct_call_to_function(self, output_file: str):
        """Save direct call instruction to function mapping to a JSON file."""
        direct_call_to_func = self.find_direct_call_instructions()
        output_data = {f"0x{call_addr:x}": f"0x{func_addr:x}" 
                      for call_addr, func_addr in direct_call_to_func.items()}
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Direct call instruction to function mapping saved to {output_file}")
        except Exception as e:
            print(f"Error saving direct call instruction to function mapping: {str(e)}")

    def save_function_to_returns(self, output_file: str):
        """Save function to return block addresses mapping to a JSON file."""
        func_to_returns = {}
        func_addrs = self.get_all_function_addresses()
        total = len(func_addrs)
        for i, func_addr in enumerate(func_addrs, 1):
            if i % 100 == 0:
                print(f"Processing function to return blocks mapping {i}/{total}...")
            func_addr_str = f"0x{func_addr:x}"
            ret_block_addrs = self.get_function_return_blocks(func_addr)
            func_to_returns[func_addr_str] = [f"0x{addr:x}" for addr in ret_block_addrs]
        if func_to_returns:
            all_return_block_set = {
                addr
                for ret_list in func_to_returns.values()
                for addr in ret_list
            }
        try:
            with open(output_file, 'w') as f:
                json.dump(func_to_returns, f, indent=2)
            print(f"Function to return blocks mapping saved to {output_file}")
        except Exception as e:
            print(f"Error saving function to return blocks mapping: {str(e)}")
            return []
            
        return all_return_block_set


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

def savefunchashes(res_folder, binary_name, retfunchashes, jtfunchashes, itcfunchashes,icfunchashes):
    ret_path = os.path.join(res_folder, f"{binary_name}_retfunchashes.json" )
    jt_path = os.path.join(res_folder, f"{binary_name}_jtfunchashes.json" )
    itc_path = os.path.join(res_folder, f"{binary_name}_itcfunchashes.json" )
    ic_path = os.path.join(res_folder, f"{binary_name}_icfunchashes.json" )

    with open(ret_path, 'w') as f:
        json.dump(retfunchashes, f, indent=4)
    with open(jt_path, 'w') as f:
        json.dump(jtfunchashes, f, indent=4)
    with open(itc_path, 'w') as f:
        json.dump(itcfunchashes, f, indent=4)
    with open(ic_path, 'w') as f:
        json.dump(icfunchashes, f, indent=4)


def _normalize_addr(value):
    """Convert angr- and JSON-style addresses into integers."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return int(value, 16) if value.startswith("0x") else int(value)
    return int(value)


def _code_node_id_for_addr(nodelookup, addr):
    """Resolve a basic-block or function-entry address to a code-node id."""
    addr = _normalize_addr(addr)
    if addr is None:
        return None

    node = nodelookup.get(addr)
    if isinstance(node, CodeTrainNode):
        return int(node.id)
    return None


def _looks_like_direct_target(insn):
    """Heuristic to separate direct control transfers from register/memory-based ones."""
    op_str = getattr(insn, "op_str", "").strip().lower()
    if not op_str:
        return False
    return op_str.startswith("0x") or op_str.startswith("-0x")


def _collect_indirect_call_source_nodes(analyzer, nodelookup):
    """Collect BB nodes containing indirect calls using angr-only call recovery."""
    all_call_to_bb = analyzer.find_all_call_instructions()
    direct_call_addrs = set(analyzer.find_direct_call_instructions().keys())
    source_nodes = set()

    for call_addr, bb_addr in all_call_to_bb.items():
        if call_addr in direct_call_addrs:
            continue
        node_id = _code_node_id_for_addr(nodelookup, bb_addr)
        if node_id is not None:
            source_nodes.add(node_id)

    return sorted(source_nodes)


def _collect_indirect_jump_source_nodes(analyzer, nodelookup):
    """Collect BB nodes ending in unresolved jmp-style dispatches."""
    source_nodes = set()

    for node in analyzer.cfg.model.nodes():
        if not node.block:
            continue

        insns = list(node.block.capstone.insns)
        if not insns:
            continue

        last_insn = insns[-1]
        if last_insn.mnemonic != "jmp":
            continue
        if _looks_like_direct_target(last_insn):
            continue

        node_id = _code_node_id_for_addr(nodelookup, node.addr)
        if node_id is not None:
            source_nodes.add(node_id)

    return sorted(source_nodes)


def _collect_return_source_nodes(analyzer, nodelookup):
    """Collect BB nodes that terminate with returns."""
    source_nodes = set()

    for func_addr in analyzer.get_all_function_addresses():
        for ret_block_addr in analyzer.get_function_return_blocks(func_addr):
            node_id = _code_node_id_for_addr(nodelookup, ret_block_addr)
            if node_id is not None:
                source_nodes.add(node_id)

    return sorted(source_nodes)


def _collect_after_call_nodes(analyzer, nodelookup):
    """Collect after-call continuation BB nodes from angr call recovery."""
    target_nodes = set()

    for next_block_addr in analyzer.analyze_all_calls_to_next_block().values():
        node_id = _code_node_id_for_addr(nodelookup, next_block_addr)
        if node_id is not None:
            target_nodes.add(node_id)

    return sorted(target_nodes)


def _collect_function_node_mappings(analyzer, nodelookup):
    """Build function-entry and function-body code-node maps."""
    function_entry_nodes = set()
    function_blocks = {}
    bb_to_func = analyzer.get_all_blocks_to_functions()

    blocks_by_function = defaultdict(set)
    for bb_addr, func_addr in bb_to_func.items():
        node_id = _code_node_id_for_addr(nodelookup, bb_addr)
        if node_id is not None:
            blocks_by_function[int(func_addr)].add(node_id)

    for func_addr in analyzer.get_all_function_addresses():
        entry_node_id = _code_node_id_for_addr(nodelookup, func_addr)
        if entry_node_id is not None:
            function_entry_nodes.add(entry_node_id)

        node_ids = blocks_by_function.get(int(func_addr), set())
        if node_ids:
            function_blocks[int(func_addr)] = sorted(node_ids)

    return sorted(function_entry_nodes), function_blocks, bb_to_func


def _collect_data_candidates(graph, task_code_candidates):
    """Collect data nodes within two undirected xRef hops of code candidates."""
    if 'data' not in graph.ntypes:
        return {task: [] for task in task_code_candidates}

    xref_relations = {
        'codexrefcode_edges',
        'codexrefdata_edges',
        'dataxrefcode_edges',
        'dataxrefdata_edges',
    }
    adjacency = defaultdict(set)
    for canonical_etype in graph.canonical_etypes:
        src_type, rel_type, dst_type = canonical_etype
        if rel_type not in xref_relations:
            continue
        src_nodes, dst_nodes = graph.edges(etype=canonical_etype)
        for src, dst in zip(
            src_nodes.detach().cpu().tolist(),
            dst_nodes.detach().cpu().tolist(),
        ):
            src_key = (src_type, int(src))
            dst_key = (dst_type, int(dst))
            adjacency[src_key].add(dst_key)
            adjacency[dst_key].add(src_key)

    data_candidates = {}
    for task, code_nodes in task_code_candidates.items():
        start_nodes = {('code', int(node_id)) for node_id in code_nodes}
        visited = set(start_nodes)
        frontier = set(start_nodes)
        task_data_nodes = set()
        for _depth in range(2):
            next_frontier = set()
            for node_key in frontier:
                for neighbor in adjacency.get(node_key, ()):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
                    if neighbor[0] == 'data':
                        task_data_nodes.add(neighbor[1])
            frontier = next_frontier
            if not frontier:
                break
        data_candidates[task] = sorted(task_data_nodes)

    return data_candidates


def build_hub_candidate_metadata(analyzer, graph, nodelookup, binary_name, binary_index):
    """Build angr-only candidate metadata for later hub augmentation."""
    function_entry_nodes, function_blocks, bb_to_func = _collect_function_node_mappings(analyzer, nodelookup)
    icall_src_nodes = _collect_indirect_call_source_nodes(analyzer, nodelookup)
    jump_src_nodes = _collect_indirect_jump_source_nodes(analyzer, nodelookup)
    ret_src_nodes = _collect_return_source_nodes(analyzer, nodelookup)
    after_call_nodes = _collect_after_call_nodes(analyzer, nodelookup)
    code_addr_by_node_id = {
        int(node.id): int(addr)
        for addr, node in nodelookup.items()
        if isinstance(node, CodeTrainNode)
    }

    jumptable_dst_by_src = {}
    for src_node_id in jump_src_nodes:
        src_addr = code_addr_by_node_id.get(int(src_node_id))
        if src_addr is None:
            continue

        func_addr = bb_to_func.get(src_addr)
        if func_addr is None:
            continue

        candidate_dst_nodes = function_blocks.get(int(func_addr), [])
        if candidate_dst_nodes:
            jumptable_dst_by_src[str(int(src_node_id))] = candidate_dst_nodes

    # Angr exposes both jump-table dispatches and indirect tail calls as
    # unresolved indirect jumps, so their source candidate sets intentionally
    # share the canonical ``jumptable_itailcall`` group.  They remain duplicated
    # under task keys for backward-compatible consumers.
    task_code_candidates = {
        "icall": sorted(set(icall_src_nodes) | set(function_entry_nodes)),
        "itailcall": sorted(set(jump_src_nodes) | set(function_entry_nodes)),
        "ret": sorted(set(ret_src_nodes) | set(after_call_nodes)),
        # Jump-table destinations do not participate in task-aware hub routing.
        "jumptable": sorted(set(jump_src_nodes)),
    }

    return {
        "binary_name": binary_name,
        "graph_index": int(binary_index),
        "candidate_source": "angr_only",
        "routing_version": "paper_v1",
        "data_candidate_radius": 2,
        "xref_distance_mode": "undirected",
        "code_candidates": {
            "src_groups": {
                "jumptable_itailcall": ["jumptable", "itailcall"],
            },
            "src": {
                "icall": icall_src_nodes,
                "itailcall": jump_src_nodes,
                "ret": ret_src_nodes,
                "jumptable": jump_src_nodes,
            },
            "dst": {
                "icall": function_entry_nodes,
                "itailcall": function_entry_nodes,
                "ret": after_call_nodes,
            },
            "dst_by_src": {
                "jumptable": jumptable_dst_by_src,
            },
        },
        "data_candidates": _collect_data_candidates(graph, task_code_candidates),
        "evaluation": {
            "long_range": {
                "icall": [],
                "itailcall": [],
                "ret": [],
                "jumptable": [],
            }
        },
    }


def save_hubmeta(res_folder, binary_name, hubmeta):
    """Persist candidate metadata next to the result-side GT files."""
    hubmeta_path = os.path.join(res_folder, f"{binary_name}_hubmeta.json")
    with open(hubmeta_path, 'w') as f:
        json.dump(hubmeta, f, indent=2)
    
def process_binary(binary_path: str, json_folder: str, sourceinfo_folder: str, resinfo_folder:str,basedir:str, graphdir:str, state,binary_index: int):
    """Process a single binary file."""
    binary_name = os.path.basename(binary_path)
    print(f"\n===== Processing binary: {binary_name} =====")
    
    # Check if the file is a valid binary
    if not is_valid_binary(binary_path):
        print(f"Error: {binary_path} is not a valid binary executable")
        return None  # Return None instead of False for invalid binaries

    # Output file paths
    funcaddr_output = os.path.join(json_folder, f"{binary_name}_funcaddr.json")
    calltoaftercall_output = os.path.join(json_folder, f"{binary_name}_calltoaftercall.json")
    functobb_output = os.path.join(json_folder, f"{binary_name}_functobb.json")
    bbtofunc_output = os.path.join(json_folder, f"{binary_name}_bbtofunc.json")
    instofunc_output = os.path.join(json_folder, f"{binary_name}_instofunc.json")
    callinstobb_output = os.path.join(json_folder, f"{binary_name}_callinstobb.json")
    jumpinstobb_output = os.path.join(json_folder, f"{binary_name}_jumpinstobb.json")
    nodeaddres_output = os.path.join(json_folder, f"{binary_name}_nodeaddres.json")
    jmpbblist_output = os.path.join(json_folder, f"{binary_name}_jumpbblist.json")
    callbblist_output = os.path.join(json_folder, f"{binary_name}_callbblist.json")
    
    # New output files
    dcinstofunc_output = os.path.join(json_folder, f"{binary_name}_dcinstofunc.json")
    functoret_output = os.path.join(json_folder, f"{binary_name}_functoret.json")
    funcascalleetocallins_output = os.path.join(json_folder, f"{binary_name}_funcascalleetodcallins.json")
    jumpinstofunc_output = os.path.join(json_folder, f"{binary_name}_jumpinstofunc.json")
    
    # Direct tail call analysis
    dtc_to_callee_output = None
    dtc_input = None
    if sourceinfo_folder:
        dtc_input = os.path.join(sourceinfo_folder, f"{binary_name}_dtcinsaddr.json")
        dtc_to_callee_output = os.path.join(sourceinfo_folder, f"{binary_name}_dtcinstocallee.json")

    try:
        print(f"Loading binary: {binary_path}")
        analyzer = AngrAnalyzer(binary_path)

        # Generate the requested outputs
        print("Generating function addresses...")
        analyzer.save_all_function_addresses(funcaddr_output)

        print("Generating call to after call mapping...")
        analyzer.save_call_to_after_call(calltoaftercall_output)

        print("Generating function to basic blocks mapping...")
        analyzer.save_function_to_blocks(functobb_output)

        print("Generating basic block to function mapping...")
        bbtofunc = analyzer.save_block_to_function(bbtofunc_output)

        print("Generating the instruction sets of first basic block to function mapping...")
        instofunc = analyzer.save_first_block_instructions_to_function(instofunc_output)
        
        print("Generating call instruction to basic block mapping...")
        analyzer.save_call_to_block(callinstobb_output)

        print("Generating jump instruction to basic block mapping...")
        analyzer.save_jump_to_block(jumpinstobb_output)
        
        print("Generating node addresses...")
        analyzer.save_node_addresses(nodeaddres_output)
        
        print("Generating jump basic blocks list...")
        analyzer.save_jump_blocks_list(jmpbblist_output)
        
        print("Generating call basic blocks list...")
        analyzer.save_call_blocks_list(callbblist_output)
        
        # New method calls
        print("Generating direct call instruction to function mapping...")
        analyzer.save_direct_call_to_function(dcinstofunc_output)
        
        print("Generating function to return blocks mapping...")
        retsource = analyzer.save_function_to_returns(functoret_output)
        
        print("Generating function (callee) to call instructions mapping...")
        analyzer.save_callee_to_call_instructions(funcascalleetocallins_output)
        
        # Add jump instruction to function mapping
        print("Generating jump instruction to function mapping...")
        analyzer.save_jump_to_function(jumpinstofunc_output)
        
        # Direct tail call analysis if sourceinfo folder is provided
        if sourceinfo_folder and os.path.exists(dtc_input):
            print("Generating direct tail call to callee function mapping...")
            analyzer.save_direct_tail_call_to_function(dtc_input, dtc_to_callee_output)

        print("Correcting jump table entries...")
        jtsource = correct_jump_mappings(binary_name, angrinfo=json_folder, sourceinfo=sourceinfo_folder, output_folder=resinfo_folder)
        
        print("Processing indirect tail call mappings...")
        itcsource = process_itc_files(binary_name, angrcfginfo=json_folder, sourceinfo=sourceinfo_folder, output_folder=resinfo_folder)

        print("Processing indirect call call mappings...")
        icsource = process_icall(binary_name, angrcfginfo=json_folder, sourceinfo=sourceinfo_folder, output_folder=resinfo_folder)
        
        retfunchashes = analyzer.get_function_hashes(retsource, bbtofunc)
        jtfunchashes = analyzer.get_function_hashes(jtsource, bbtofunc)
        itcfunchashes = analyzer.get_function_hashes(itcsource, bbtofunc)
        icfunchashes = analyzer.get_function_hashes(icsource, bbtofunc)


        savefunchashes(resinfo_folder, binary_name, retfunchashes, jtfunchashes, itcfunchashes,icfunchashes)
        
        print(f"Analysis complete for {binary_name}!")
        print("build graph")
        g, nodelookup= analyzer.build_graph(binary_path,state)
        analyzer.save_graph(binary_name,json_folder,graphdir,g,nodelookup,binary_index)
        hubmeta = build_hub_candidate_metadata(analyzer, g, nodelookup, binary_name, binary_index)
        save_hubmeta(resinfo_folder, binary_name, hubmeta)
        explicit_cf_edge_types = ['codecall_edges', 'codejump_edges']
        total_cf_edge_types = ['codecall_edges', 'codejump_edges', 'code2code_edges']

        graph_stats = {
            "binary": binary_name,
            "code_nodes": g.num_nodes('code'),
            "data_nodes": g.num_nodes('data'),
            "total_functions": len(analyzer.get_all_function_addresses()),
            "total_edges": sum(g.num_edges(et) for et in g.etypes),  # Optional: keep or remove
            "explicit_control_flow_edges": sum(
                g.num_edges(et) for et in explicit_cf_edge_types if et in g.etypes
            ),
            "total_control_flow_edges": sum(
                g.num_edges(et) for et in total_cf_edge_types if et in g.etypes
            ),
            "total_xref_cf_edge": sum(
                g.num_edges(et) for et in g.etypes
            ),
            "edges_involving_data": sum(
                g.num_edges(et) for et in g.canonical_etypes
                if et[0] == 'data' or et[2] == 'data'
            )
        }
        graph_stat_output = os.path.join(resinfo_folder, f"{binary_name}_graphstats.json")
        with open(graph_stat_output, 'w') as f:
            json.dump(graph_stats, f, indent=2)
        print(f"Graph statistics saved to: {graph_stat_output}")
        return True
    except Exception as e:
        print(f"Error during analysis of {binary_name}: {str(e)}")
        return False

def main():
    
    state = GlobalState()
    parser = argparse.ArgumentParser(description='Generate binary analysis data using angr')
    parser.add_argument('binary', help='Folder containing binary files')
    parser.add_argument('angrcfginfo', help='Output folder for analysis results')
    parser.add_argument('sourceinfo', help='Folder containing source information files')
    parser.add_argument('resinfo', help='Folder storaging res info')
    parser.add_argument('basedir', help='Folder containing source information files')
    parser.add_argument('binary_index',type=int, help='Folder containing source information files')
    args = parser.parse_args()
    
    # Create output folder if it doesn't exist
    os.makedirs(args.angrcfginfo, exist_ok=True)
    os.makedirs(args.basedir,exist_ok=True)
    graphdir = os.path.join(args.basedir, "graph")
    os.makedirs(graphdir, exist_ok=True)
    bintoindex_path = os.path.join(args.basedir, "bintoindex.json")
    if not os.path.exists(bintoindex_path):
        print(f"{bintoindex_path} does not exist. Creating a new one.")
        bintoindex = {}  # or any default value
    else:
        with open(bintoindex_path, 'r') as f:
            bintoindex = json.load(f)
    
    indextores_path = os.path.join(args.basedir, "indextores.json")
    if not os.path.exists(indextores_path):
        print(f"{indextores_path} does not exist. Creating a new one.")
        indextores = {}  # or any default value
    else:
        with open(indextores_path, 'r') as f:
            indextores = json.load(f)
    # Validate sourceinfo folder if provided
    if args.sourceinfo and not os.path.isdir(args.sourceinfo):
        print(f"Error: Sourceinfo folder {args.sourceinfo} not found or not a directory")
        sys.exit(1)
    
    # Get list of all files in the binary folder
    binary_path = args.binary
    if binary_path in bintoindex:
        print(f"{binary_path} is already processed, skipping.")
        return  # or continue inside a loop
    
    result = process_binary(binary_path, args.angrcfginfo, args.sourceinfo, args.resinfo,args.basedir, graphdir, state, args.binary_index)
    
    if result:
        bintoindex[binary_path] = args.binary_index
        indextores[args.binary_index] = args.resinfo
    
        with open(bintoindex_path, 'w') as f:
            json.dump(bintoindex, f, indent=4)
    
        with open(indextores_path, 'w') as f:
            json.dump(indextores,f, indent=4)
    else: 
        sys.exit(1)



if __name__ == "__main__":
    main()
