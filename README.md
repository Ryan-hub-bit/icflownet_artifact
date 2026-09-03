# ICFlowNet minimal artifact

This repository contains the smallest reviewer-facing workflow for the three
experiments in the ICFlowNet Artifact Appendix:

- **E1 / C1:** static and dynamic ICF ground truth plus heterogeneous ACFG
  construction, executed in the released Docker image;
- **E2 / C2:** one-epoch Dual-Hub training smoke test and the released
  Dual-Hub versus No-Hub indirect-call comparison; and
- **E3 / C3:** one-epoch four-task training smoke test and the released MTL
  versus Dual-Hub single-task indirect-call comparison.

The 20-graph training runs check functionality only. They do not retrain the
released models or reproduce full-corpus training. Reported model comparisons
always use the released checkpoints, the same clean-test pairs, a fixed
threshold of `0.5`, and FP32 inference.

## Artifact access

- **Source code:**
  [Ryan-hub-bit/icflownet_artifact](https://github.com/Ryan-hub-bit/icflownet_artifact)
- **Functional AE data and released models:**
  [Zenodo 10.5281/zenodo.22262200](https://doi.org/10.5281/zenodo.22262200)
  (`icflownet-ae.tar.gz`)
- **Optional full raw-binary dataset:**
  [Zenodo 10.5281/zenodo.22261977](https://doi.org/10.5281/zenodo.22261977)
  (`binary.tar.gz`)
- **E1 Docker image:**
  `ghcr.io/ryan-hub-bit/icflow-dynamic-collection:e1`

Only the functional AE package and Docker image are required for E1--E3. The
full raw-binary dataset is provided for reuse and full-corpus regeneration.

## Requirements

E2 and E3 were tested on x86-64 Linux with Conda, CUDA 11.8, and two visible
CUDA GPUs. E2 training and single-task evaluation use one GPU. E3 training uses
one GPU and its distributed clean-test evaluation uses exactly two GPU
processes. The paper experiments used NVIDIA A100 80 GB GPUs.

E1 requires Docker, Internet access, an x86-64 Linux host, approximately 25 GB
of free disk, and no GPU. The container needs `SYS_PTRACE` and an unconfined
seccomp profile for Pin. The Docker daemon must be running, and the user must
have direct Docker access or permission to run Docker with `sudo`.

The model and static-analysis source tree is retained for inspection and reuse.
Full-corpus training and dataset regeneration remain outside the timed
functional workflow because the minimal AE data package does not contain the
complete raw corpus.

## Repository and data layout

The reviewer-facing entry points are isolated under `AE_scripts/`; the complete
ICFlowNet implementation remains under `src/`:

```text
icflownet_artifact/
├── AE_scripts/                 # preflight, E2/E3 runners, and verifiers
├── scripts/                    # clean-test evaluators and static-GT pipeline
├── src/                        # complete model, training, GT, and graph source
├── environment.yml            # host training/evaluation environment
├── graphenv.yml                # graph-generation dependency record
└── README.md
```

The separately downloaded AE data archive must extract to this layout:

```text
ae_data/
├── model/
│   ├── ic_hub_model.pt
│   ├── ic_nohub_model.pt
│   └── mtl_model.pt
├── cleantest/
│   ├── binary_graph_mapping.jsonl
│   ├── binaries/
│   ├── graphs/
│   ├── pairs/
│   └── resources/
└── train_sample/
    ├── binary/
    ├── graph/
    ├── res/
    └── portable JSON indexes
```

The package contains clean-test data and a small training-smoke sample. The
training sample covers all four tasks: `ret`, `jumptable`, `indirectcall`, and
`tailcall`.

## 1. Setup and basic test

Clone the repository and extract the functional AE package:

```bash
git clone \
  https://github.com/Ryan-hub-bit/icflownet_artifact.git

cd icflownet_artifact

export REPO_ROOT="$PWD"
export OUTPUT_ROOT="$REPO_ROOT/outputs"
export AE_SCRIPTS="$REPO_ROOT/AE_scripts"

mkdir -p "$REPO_ROOT/ae_data" "$OUTPUT_ROOT"

wget -O icflownet-ae.tar.gz \
  "https://zenodo.org/records/22262200/files/icflownet-ae.tar.gz?download=1"

tar --strip-components=1 \
  -xzf icflownet-ae.tar.gz \
  -C "$REPO_ROOT/ae_data"

export DATA_ROOT="$REPO_ROOT/ae_data"
```

Create and activate the pinned environment:

```bash
conda env create --name icflownet-ae --file environment.yml
conda activate icflownet-ae
```

Select an available Docker command, pull the E1 image, and check that its
experiment runner is present:

```bash
export IMG=ghcr.io/ryan-hub-bit/icflow-dynamic-collection:e1

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

"${DOCKER[@]}" info >/dev/null
"${DOCKER[@]}" pull "$IMG"

"${DOCKER[@]}" run --rm "$IMG" bash -lc '
test -f /opt/icflownet-e1/run_e1.sh &&
echo "E1 container PASS"
'
```

The commands use ordinary Docker when the current user can access the daemon;
otherwise, they use `sudo docker`, which may prompt for a password. If neither
method is authorized, ask the host administrator for Docker access before
running E1.

Expected output:

```text
E1 container PASS
```

Run the host-environment basic test. The script performs the preflight and a
four-record, two-GPU MTL evaluation:

```bash
bash "$AE_SCRIPTS/run_smoke.sh"
```

Expected final lines:

```text
Environment PASS: ...
Artifact data PASS: ...
AE preflight PASS
Basic evaluation PASS
```

The preflight checks dependency versions, two-GPU visibility, package paths,
record counts, task coverage, checkpoint metadata, and SHA-256 checksums.

## 2. E1: ground truth and graph generation

E1 evaluates claim C1 from Sections 4.1–4.2: ICFlowNet supports static and
dynamic indirect-control-flow ground-truth generation and angr-based
heterogeneous ACFG construction from a stripped binary. The experiment uses
the real `ZydisInfo` ELF built from the pinned Zydis package. It takes about 10
person-minutes to launch, up to one compute-hour, and approximately 25 GB of
disk space.

E1 runs entirely in the Docker image pulled during setup. It does not use the
host Conda environment or require a GPU. Run:

```bash
"${DOCKER[@]}" rm -f icflownet-e1 >/dev/null 2>&1 || true

"${DOCKER[@]}" run --name icflownet-e1 --init \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  "$IMG" \
  /opt/icflownet-e1/run_e1.sh \
  --sample zydis \
  --output /output/zydis
```

The published Linux/amd64 image manifest used for the artifact is pinned by
digest `sha256:665682fd5f5e7185d3e76dcedea4b6d7406b12dca8c77f1f3dffc164136d65de`.

The runner performs five automatic stages:

1. **Compile the pinned Zydis package with the custom LLVM toolchain.**
   Expected: the `ZydisInfo` ELF is created and the package regression test
   passes.
2. **Generate dynamic and static ground truth.** Pin/MyPinTool records dynamic
   indirect control-flow during the local package test, while compiler labels
   provide static GT. Expected: all 77 dynamically observed ICF pairs are
   covered by the 93 static GT pairs.
3. **Strip a copy of the binary.** Expected: symbols are removed while the
   `.text` section bytes remain unchanged.
4. **Construct the heterogeneous ACFG from the stripped copy.** angr recovers
   the graph and DGL serializes it. Expected: 546 code nodes, 249 data nodes,
   and 2,273 edges.
5. **Run the automatic verifier.** Expected: every assertion passes, the
   process exits with status zero, and the final output is:

```text
E1 PASS: Zydis static GT, dynamic GT, and stripped-binary heterogeneous ACFG verified
```

Any compilation, package-test, ground-truth coverage, stripping, graph-shape,
or verification failure causes a nonzero exit status.

Collect the evidence and remove the stopped container:

```bash
mkdir -p "$OUTPUT_ROOT"
"${DOCKER[@]}" cp icflownet-e1:/output/zydis "$OUTPUT_ROOT/e1"
"${DOCKER[@]}" rm icflownet-e1
cat "$OUTPUT_ROOT/e1/verification.json"
```

`verification.json` must contain `"status": "PASS"`. The copied directory also
contains the labeled, dynamic-GT, stripped-binary, static-GT, graph, build-log,
test-log, proof, and run-log evidence used by the verifier. The label-derived
static GT is extracted before stripping; graph construction then uses the
stripped copy whose `.text` hash must match the labeled ELF.

## 3. E2: Dual Virtual Hub ablation

Run E2 from the activated Conda environment:

```bash
cd "$REPO_ROOT"
bash "$AE_SCRIPTS/run_e2.sh"
```

The script trains a Dual-Hub indirect-call model for one epoch on all 20
eligible smoke graphs, evaluates the released Hub and No-Hub checkpoints on
both clean-test scopes in FP32, and verifies matching data, pair counts,
thresholds, precision, and checkpoint configurations.

Expected result:

```text
overall: Hub F1 > No-Hub F1
long_range: Hub F1 > No-Hub F1
E2 PASS
```

Results are written under `outputs/e2/`. The smoke checkpoint is not used in
the reported comparison. `E2 PASS` is printed only when Dual-Hub F1 is higher
than No-Hub F1 on both scopes. To rerun only the automatic verifier:

```bash
python "$AE_SCRIPTS/verify_e2.py" "$OUTPUT_ROOT/e2"
```

## 4. E3: multi-task comparison

E3 consumes E2's released Dual-Hub results, so run E2 first:

```bash
cd "$REPO_ROOT"
bash "$AE_SCRIPTS/run_e3.sh"
```

The script trains the four-task MTL architecture for one epoch on the 20-graph
sample, evaluates the released MTL checkpoint on `overall` and `long_range`
using two GPU processes and FP32, and compares its indirect-call result with
the released Dual-Hub single-task checkpoint.

Expected result:

```text
overall: MTL F1 > Single F1
long_range: MTL F1 > Single F1
E3 PASS
```

Results are written under `outputs/e3/`. The MTL smoke checkpoint is not used
in the reported comparison. `E3 PASS` is printed only when MTL indirect-call
F1 is higher than single-task F1 on both scopes.

The E3 verifier can be rerun independently:

```bash
python "$AE_SCRIPTS/verify_e3.py" \
  "$OUTPUT_ROOT/e2" \
  "$OUTPUT_ROOT/e3"
```

It prints `E3 PASS` only when both evaluations use the same clean-test root,
indirect-call pair counts, threshold, checkpoint format, and FP32 mode, and MTL
indirect-call F1 is higher in both scopes.

## Troubleshooting

- Do not enable AMP for artifact results. FP16 autocast can produce a `NaN`
  return-task loss. FP32 is the default; `--disable_amp` remains accepted only
  for explicitness and compatibility.
- If NCCL initialization fails because of a local driver/NVML mismatch, rerun
  `run_smoke.sh` or `run_e3.sh` with GPU inference unchanged and Gloo used only
  for metric reduction:

  ```bash
  export ICFLOWNET_DIST_BACKEND=gloo
  ```

- The artifact runners never fall back to CPU inference. Verify that two GPUs
  are visible with `python "$AE_SCRIPTS/check_setup.py" "$DATA_ROOT"`.
- To use a different training GPU for E2 and the E3 smoke-training stage, set
  `GPU`, for example `export GPU=1`. Distributed E3 evaluation still uses the
  two devices visible to `torchrun`.

## Output interpretation

E1 validates generation functionality. E2 and E3 training losses are smoke
results only. The claims are accepted only by the automatic verifiers using
the released checkpoints and clean-test pairs. Any missing file, altered
checkpoint, mismatched pair count/root/threshold/precision, wrong Hub
configuration, or failed F1 inequality causes a nonzero exit status.
