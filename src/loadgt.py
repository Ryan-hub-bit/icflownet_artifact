"""Load ground-truth edge annotations for a graph index and split mode."""

import os 
import json
from common import EDGE_TYPE_MAPPING
# Constants and configuration
TRETEX = '_ret.json'  # File extension for JSON files
TJTEX = '_correctjumptable.json'  # File extension for JSON files
TTCEX = '_itcbbtofunc.json'
TICEX = '_icallbbtocallee.json'
VRETEX = '_ret_eval.json'  # File extension for JSON files
VJTEX = '_correctjumptable_eval.json'  # File extension for JSON files
VTCEX = '_itcbbtofunc_eval.json'
VICEX = '_icallbbtocallee_eval.json'
VRETEX_DU = '_ret_eval_duplication.json'  # File extension for JSON files
VJTEX_DU = '_correctjumptable_eval_duplication.json'  # File extension for JSON files
VTCEX_DU = '_itcbbtofunc_eval_duplication.json'
VICEX_DU = '_icallbbtocallee_eval_duplication.json'
NODELOOKUPEX = '_nodelookup.json'


def loadgt(basedir, state, mode):   
    indextores_path = os.path.join(basedir, "indextores.json")
    with open(indextores_path, 'r') as f:
        json_raw = f.read()
        indextores = json.loads(json_raw)
    gt_folder = indextores[state]
    binary_name = os.path.basename(gt_folder)
    duplicate_gt = []
    duplicate_type = []
    nodelookup_path = os.path.join(gt_folder, binary_name + NODELOOKUPEX)
 
    with open(nodelookup_path, 'r') as f:
        node_lookup = json.load(f)  # directly loads the dict
    #print(len(node_lookup))
    def load_json_or_empty(path):
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            json_raw = f.read()
            return json.loads(json_raw)

    def eval_or_train_path(eval_ext, train_ext):
        eval_path = os.path.join(gt_folder, binary_name + eval_ext)
        if os.path.exists(eval_path):
            return eval_path
        return os.path.join(gt_folder, binary_name + train_ext)

    if mode == "train" or mode =="allowduplicate":
        jt_path = os.path.join(gt_folder, binary_name + TJTEX)
        ic_path = os.path.join(gt_folder, binary_name + TICEX)
        tc_path = os.path.join(gt_folder, binary_name + TTCEX)
        ret_path = os.path.join(gt_folder, binary_name + TRETEX)
    elif mode == "eval":
        jt_path = eval_or_train_path(VJTEX, TJTEX)
        ic_path = eval_or_train_path(VICEX, TICEX)
        tc_path = eval_or_train_path(VTCEX, TTCEX)
        ret_path = eval_or_train_path(VRETEX, TRETEX)
        jt_path_du = os.path.join(gt_folder, binary_name + VJTEX_DU)
        ic_path_du= os.path.join(gt_folder, binary_name + VICEX_DU)
        tc_path_du = os.path.join(gt_folder, binary_name + VTCEX_DU)
        ret_path_du= os.path.join(gt_folder, binary_name + VRETEX_DU)
        jt_data_du = load_json_or_empty(jt_path_du)
        ic_data_du = load_json_or_empty(ic_path_du)
        tc_data_du = load_json_or_empty(tc_path_du)
        ret_data_du = load_json_or_empty(ret_path_du)
        
        jt_edges_du = []
        ic_edges_du = []
        tc_edges_du = []
        ret_edges_du = []     
        if isinstance(jt_data_du, dict):
            for key, vals in jt_data_du.items():   
                nodeid = node_lookup.get(key, -1)
                if nodeid == -1:
                    continue
                for val in vals:
                    tnodeid = node_lookup.get(val, -1)
                    if tnodeid == -1:
                        continue
                    jtedge_du = (nodeid, tnodeid)
                    jt_edges_du.append(jtedge_du)
    
        if isinstance(tc_data_du, dict):
            for key, vals in tc_data_du.items():
                nodeid = node_lookup.get(key, -1)
                if nodeid == -1:
                    continue
                for val in vals:
                    tnodeid = node_lookup.get(val, -1)
                    if tnodeid == -1:
                        continue
                    tcedge_du = (nodeid, tnodeid)
                    tc_edges_du.append(tcedge_du)

        if isinstance(ic_data_du, dict):
            for key, vals in ic_data_du.items():
                nodeid = node_lookup.get(key, -1)
                if nodeid == -1:
                    continue
                for val in vals:
                    tnodeid = node_lookup.get(val, - 1)
                    if tnodeid == -1:
                        continue
                    icedge_du = (nodeid, tnodeid)
                    ic_edges_du.append(icedge_du)
    
        if isinstance(ret_data_du, dict):
            for key, vals in ret_data_du.items():
                nodeid = node_lookup.get(key, -1)
                if nodeid == -1:
                    continue
                for val in vals:
                    tnodeid = node_lookup.get(val, -1)
                    if tnodeid == -1:
                        continue
                    retedge_du = (nodeid, tnodeid)  # Fixed variable name (was using icedge)
                    ret_edges_du.append(retedge_du)  # Fixed to use ret_edges
        duplicate_gt = jt_edges_du + tc_edges_du + ic_edges_du + ret_edges_du
        for _ in jt_edges_du:
            duplicate_type.append(EDGE_TYPE_MAPPING["jumptable"])
        for _ in tc_edges_du:
            duplicate_type.append(EDGE_TYPE_MAPPING["tailcall"])
        for _ in ic_edges_du:
            duplicate_type.append(EDGE_TYPE_MAPPING["indirectcall"])
        for _ in ret_edges_du:
            duplicate_type.append(EDGE_TYPE_MAPPING["ret"])
        
        


        

    # Read and parse JSON
    with open(jt_path, 'r') as f:
        json_raw = f.read()
        jt_data = json.loads(json_raw)
    with open(ic_path, 'r') as f:
        json_raw = f.read()
        ic_data = json.loads(json_raw)
    with open(tc_path, 'r') as f:
        json_raw = f.read()
        tc_data = json.loads(json_raw)
    with open(ret_path, 'r') as f:
        json_raw = f.read()
        ret_data = json.loads(json_raw)

        
    jt_edges = []
    ic_edges = []
    tc_edges = []
    ret_edges = []     
        
    # Ground truth
    if isinstance(jt_data, dict):
        for key, vals in jt_data.items():   
            nodeid = node_lookup.get(key, -1)
            if nodeid == -1:
                continue
            for val in vals:
                tnodeid = node_lookup.get(val, -1)
                if tnodeid == -1:
                    continue
                jtedge = (nodeid, tnodeid)
                jt_edges.append(jtedge)
    
    if isinstance(tc_data, dict):
        for key, vals in tc_data.items():
            nodeid = node_lookup.get(key, -1)
            if nodeid == -1:
                continue
            for val in vals:
                tnodeid = node_lookup.get(val, -1)
                if tnodeid == -1:
                    continue
                tcedge = (nodeid, tnodeid)
                tc_edges.append(tcedge)

    if isinstance(ic_data, dict):
        for key, vals in ic_data.items():
            nodeid = node_lookup.get(key, -1)
            if nodeid == -1:
                continue
            for val in vals:
                tnodeid = node_lookup.get(val, - 1)
                if tnodeid == -1:
                    continue
                icedge = (nodeid, tnodeid)
                ic_edges.append(icedge)
    
    if isinstance(ret_data, dict):
        for key, vals in ret_data.items():
            nodeid = node_lookup.get(key, -1)
            if nodeid == -1:
                continue
            for val in vals:
                tnodeid = node_lookup.get(val, -1)
                if tnodeid == -1:
                    continue
                retedge = (nodeid, tnodeid)  # Fixed variable name (was using icedge)
                ret_edges.append(retedge)  # Fixed to use ret_edges

    # Combine all ground truth edges
    gt_type = []
    gt_edges = jt_edges + tc_edges + ic_edges + ret_edges
    # Assign types to each edge
    for _ in jt_edges:
        gt_type.append(EDGE_TYPE_MAPPING["jumptable"])
    for _ in tc_edges:
        gt_type.append(EDGE_TYPE_MAPPING["tailcall"])
    for _ in ic_edges:
        gt_type.append(EDGE_TYPE_MAPPING["indirectcall"])
    for _ in ret_edges:
        gt_type.append(EDGE_TYPE_MAPPING["ret"])
    return gt_edges, gt_type, duplicate_gt, duplicate_type
