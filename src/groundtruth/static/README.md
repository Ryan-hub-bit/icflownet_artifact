# `src/groundtruth/static`

Static graph and ground-truth generation pipeline. The recommended public entry
point is `scripts/getstaticgt_pipeline.sh`; individual Python files are usually
called by that shell script in order.

## Main Flow

1. `parsebylabel.py` reads label metadata from an ELF binary and emits
   source/target JSON files under `sourceinfo/<binary>/`.
2. `angrcfginfo.py` recovers the CFG with angr, builds the heterogeneous DGL
   graph, writes `graph/<idx>.graph.gz`, and emits lookup/metadata files.
3. `process*` and `get*` scripts normalize labels from instruction addresses
   to basic-block node keys.
4. `mergeret.py` merges return supervision.
5. `getindextobin.py` creates `indextobin.json` from `bintoindex.json`.

## Important Outputs

For each binary, the pipeline writes these files under `res/<binary>/`:

- `<binary>_nodelookup.json`: address to graph node id.
- `<binary>_hubmeta.json`: candidate metadata for hub routing. Its
  `code_candidates.src_groups.jumptable_itailcall` marker records that angr
  cannot reliably distinguish jump-table and indirect-tail-call source sites.
  Both tasks therefore share source candidates; task-aware routing keeps those
  source-to-hub edges but omits jump-table destination-to-hub edges.
- `<binary>_icallbbtocallee.json`: indirect-call GT.
- `<binary>_itcbbtofunc.json`: tail-call GT.
- `<binary>_correctjumptable.json`: jump-table GT.
- `<binary>_ret.json`: return GT.
- `<binary>_graphstats.json`: graph construction statistics.

## Script Index

- `angrcfginfo.py`: CPU/static CFG extractor and graph builder.
- `parsebylabel.py`: label parser and source/target JSON writer.
- `processjumptable.py`: repairs jump-table sources to basic-block starts.
- `processtailcall.py`: maps indirect tail-call labels to basic blocks.
- `processicall.py`: maps indirect-call labels to basic blocks.
- `processdcallreturn.py`: direct-call return mapping.
- `processicallreturn.py`: indirect-call return mapping.
- `getfuncascalleetoicallins.py`: inverts icall mappings into callee-to-callsite
  form.
- `combinefuncascalleetocallins.py`: merges direct and indirect callee-to-callsite
  mappings.
- `gettcfunctocallee.py`: converts tail-call mappings into function-to-callee
  mappings.
- `gettcfuncallpathret.py`: follows tail-call chains and collects return blocks.
- `getrettoaftercallfortc.py`: links tail-call return blocks to after-call
  destinations.
- `mergeret.py`: final return-supervision merger.
- `getindextobin.py`: creates `indextobin.json`.
- `eval_utils.py`, `vocab.py`, `config.py`, `logger.py`: PalmTree embedding and
  utility support.
- `palmtree/`: bundled PalmTree vocabulary and checkpoint files.
