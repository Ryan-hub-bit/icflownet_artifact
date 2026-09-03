# `src/groundtruth`

Ground-truth generation code for turning instrumented/labeled binaries into
DGL graph files and static per-edge-type supervision JSONs.

## Folders

- `static/`: the main reproducible pipeline for graph and static GT generation.
  It is orchestrated by `scripts/getstaticgt_pipeline.sh`.

The static pipeline follows this high-level structure:

1. Parse label metadata.
2. Recover CFG/basic-block structure.
3. Emit a DGL graph and node lookup files.
4. Normalize per-type supervision into JSON files consumed by `src/loadgt.py`.

Dynamic GT is distributed with the released data and is read by the evaluation
scripts; its generation pipeline is not included in this anonymous artifact.

The generated dataset root is expected to contain `graph/`, `res/`,
`bintoindex.json`, `indextores.json`, and `indextobin.json`.
