"""Helpers for removing train/validation leakage via function-hash based filtering."""

import os
import json
from collections import Counter


FUNCTION_HASH_SUFFIXES = (
    "_jtfunchashes.json",
    "_retfunchashes.json",
    "_icfunchashes.json",
    "_itcfunchashes.json",
)


def count_function_hash_sidecars(indices, indextores):
    """Count available function-hash sidecars for the selected graph indices."""
    count = 0
    for indice in set(str(value) for value in indices):
        res_folder = indextores.get(indice)
        if not res_folder:
            continue
        basefolder = os.path.basename(res_folder)
        count += sum(
            os.path.exists(os.path.join(res_folder, f"{basefolder}{suffix}"))
            for suffix in FUNCTION_HASH_SUFFIXES
        )
    return count

#Need to be tested 
# def extract_funchash_sets_multi(indice: int, indextores: dict):
#     fn_set = set()
#     res_folder = indextores[str(indice)]
#     basefolder = os.path.basename(res_folder)
#     tjson = os.path.join(res_folder, f"{basefolder}_funchashes.json")

#     if os.path.exists(tjson):
#         with open(tjson, 'r') as f:
#             data = json.load(f)
#             fn_set.update(data.values())  # Get only function hashes (not source keys)
#     else:
#         print(f"[WARNING] Missing funchash file: {tjson}")

#     return fn_set

def extract_funchash_sets(indice: int, indextores: dict):
    """
    Given an index and the index-to-result-folder mapping, extract the sets of function hashes
    used in jump table (JT), return (RET), indirect call (ICALL), and indirect tail call (ITC) sites.

    Args:
        indice (int): The index of the graph.
        indextores (dict): Mapping from index to result folder path.
        basedir (str): Base directory containing all result folders.

    Returns:
        tuple[set, set, set, set]: A tuple of sets (jt_set, ret_set, icall_set, itc_set).
    """
    jt_set, ret_set, icall_set, itc_set = set(), set(), set(), set()


    res_folder = indextores[str(indice)]
    basefolder = os.path.basename(res_folder)

    def load_hash_set(site_file, hash_file):
        result = set()
        if os.path.exists(site_file) and os.path.exists(hash_file):
            with open(site_file, 'r') as sf, open(hash_file, 'r') as hf:
                site_data = json.load(sf)
                hash_data = json.load(hf)
                for site in site_data:
                    hashval = hash_data.get(site)
                    if hashval:
                        result.add(hashval)
        return result

    jt_json = os.path.join(res_folder, f"{basefolder}_correctjumptable.json")
    jt_hash_json = os.path.join(res_folder, f"{basefolder}_jtfunchashes.json")
    jt_set = load_hash_set(jt_json, jt_hash_json)

    ret_json = os.path.join(res_folder, f"{basefolder}_ret.json")
    ret_hash_json = os.path.join(res_folder, f"{basefolder}_retfunchashes.json")
    ret_set = load_hash_set(ret_json, ret_hash_json)

    icall_json = os.path.join(res_folder, f"{basefolder}_icallbbtocallee.json")
    icall_hash_json = os.path.join(res_folder, f"{basefolder}_icfunchashes.json")
    icall_set = load_hash_set(icall_json, icall_hash_json)

    itc_json = os.path.join(res_folder, f"{basefolder}_itcbbtofunc.json")
    itc_hash_json = os.path.join(res_folder, f"{basefolder}_itcfunchashes.json")
    itc_set = load_hash_set(itc_json, itc_hash_json)

    return jt_set, ret_set, icall_set, itc_set


# def filter_and_save_val_jsons_multi(trainfnhash, indice, indextores,counter=None):
#     res_folder = indextores[str(indice)]
#     basefolder = os.path.basename(res_folder)
#     log_path = os.path.join(res_folder, f"{basefolder}_allval_filter.log")
#     log_lines = []
#     def filter_and_save(site_file, hash_file, hash_set, suffix):
#         if not os.path.exists(site_file) or not os.path.exists(hash_file):
#             msg = f"[{suffix.upper()}] Index {indice}: MISSING file(s). Skipping."
#             print(msg)
#             log_lines.append(msg)
#             return

#         with open(site_file, 'r') as f_site, open(hash_file, 'r') as f_hash:
#             site_data = json.load(f_site)
#             hash_data = json.load(f_hash)

#         total_entries = len(site_data)
#         kept = {}
#         removed = {}

#         for k, v in site_data.items():
#             h = hash_data.get(k)
#             if h in hash_set:
#                 removed[k] = v
#             else:
#                 kept[k] = v

#         kept_path = os.path.join(res_folder, f"{basefolder}_{suffix}_multieval.json")
#         removed_path = os.path.join(res_folder, f"{basefolder}_{suffix}_multieval_duplication.json")

#         try:
#             with open(kept_path, 'w') as f_out:
#                 json.dump(kept, f_out, indent=2)
#             with open(removed_path, 'w') as f_out_dup:
#                 json.dump(removed, f_out_dup, indent=2)
#         except Exception as e:
#             print(f"❌ Failed to write output files for {suffix}: {e}")

#         msg = f"[{suffix.upper()}] Index {indice}: total = {total_entries}, removed = {len(removed)}, kept = {len(kept)}"
#         log_lines.append(msg)

#         if counter:
#             counter(total_entries, len(removed))
#     # Process each site type
#     filter_and_save(
#         os.path.join(res_folder, f"{basefolder}_correctjumptable.json"),
#         os.path.join(res_folder, f"{basefolder}_funchashes.json"),
#         trainfnhash,
#         "correctjumptable"
#     )
#     filter_and_save(
#         os.path.join(res_folder, f"{basefolder}_ret.json"),
#         os.path.join(res_folder, f"{basefolder}_retfunchashes.json"),
#         trainfnhash,
#         "ret"
#     )
#     filter_and_save(
#         os.path.join(res_folder, f"{basefolder}_icallbbtocallee.json"),
#         os.path.join(res_folder, f"{basefolder}_icfunchashes.json"),
#         trainfnhash,
#         "icallbbtocallee"
#     )
#     filter_and_save(
#         os.path.join(res_folder, f"{basefolder}_itcbbtofunc.json"),
#         os.path.join(res_folder, f"{basefolder}_itcfunchashes.json"),
#         trainfnhash,
#         "itcbbtofunc"
#     )

#     # Write logs to file
#     with open(log_path, 'w') as log_file:
#         log_file.write("\n".join(log_lines) + "\n")
            
def filter_and_save_val_jsons(
    jt_hashes, ret_hashes, icall_hashes, itc_hashes, indice, indextores, 
    counter=None, 
    global_counter=None  # <-- new parameter!
):
    """
    Filter per-index AND contribute removed hash counts to a global counter.
    """
    res_folder = indextores[str(indice)]
    basefolder = os.path.basename(res_folder)
    log_path = os.path.join(res_folder, f"{basefolder}_val_filter.log")
    log_lines = []

    # Per-index counter
    removed_hash_counter = Counter()

    def filter_and_save(site_file, hash_file, hash_set, suffix):
        if not os.path.exists(site_file) or not os.path.exists(hash_file):
            msg = f"[{suffix.upper()}] Index {indice}: MISSING file(s). Skipping."
            print(msg)
            log_lines.append(msg)
            return

        with open(site_file, 'r') as f_site, open(hash_file, 'r') as f_hash:
            site_data = json.load(f_site)
            hash_data = json.load(f_hash)

        total_entries = len(site_data)
        kept = {}
        removed = {}

        for k, v in site_data.items():
            h = hash_data.get(k)
            if h in hash_set:
                removed[k] = v
                if h:
                    removed_hash_counter[h] += 1
            else:
                kept[k] = v

        kept_path = os.path.join(res_folder, f"{basefolder}_{suffix}_eval.json")
        removed_path = os.path.join(res_folder, f"{basefolder}_{suffix}_eval_duplication.json")

        with open(kept_path, 'w') as f_out:
            json.dump(kept, f_out, indent=2)
        with open(removed_path, 'w') as f_out_dup:
            json.dump(removed, f_out_dup, indent=2)

        msg = f"[{suffix.upper()}] Index {indice}: total = {total_entries}, removed = {len(removed)}, kept = {len(kept)}"
        log_lines.append(msg)

        if counter:
            counter(total_entries, len(removed))

    # Process each type
    filter_and_save(
        os.path.join(res_folder, f"{basefolder}_correctjumptable.json"),
        os.path.join(res_folder, f"{basefolder}_jtfunchashes.json"),
        jt_hashes,
        "correctjumptable"
    )
    filter_and_save(
        os.path.join(res_folder, f"{basefolder}_ret.json"),
        os.path.join(res_folder, f"{basefolder}_retfunchashes.json"),
        ret_hashes,
        "ret"
    )
    filter_and_save(
        os.path.join(res_folder, f"{basefolder}_icallbbtocallee.json"),
        os.path.join(res_folder, f"{basefolder}_icfunchashes.json"),
        icall_hashes,
        "icallbbtocallee"
    )
    filter_and_save(
        os.path.join(res_folder, f"{basefolder}_itcbbtofunc.json"),
        os.path.join(res_folder, f"{basefolder}_itcfunchashes.json"),
        itc_hashes,
        "itcbbtofunc"
    )

    # Save per-index log
    with open(log_path, 'w') as log_file:
        log_file.write("\n".join(log_lines) + "\n")

    # Aggregate into global counter if provided
    if global_counter is not None:
        global_counter.update(removed_hash_counter)
