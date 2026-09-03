#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_ROOT/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
IMAGE="${ICFLOWNET_E1_IMAGE:-ghcr.io/ryan-hub-bit/icflow-dynamic-collection:e1}"
CONTAINER="${ICFLOWNET_E1_CONTAINER:-icflownet-e1}"
RESULT_ROOT="${ICFLOWNET_E1_OUTPUT:-$OUTPUT_ROOT/e1}"
CHECK_ONLY=false

usage() {
    cat <<'EOF'
Usage: run_e1_host.sh [--check-only]

Without options, run E1 and copy its verified evidence to outputs/e1.
With --check-only, only verify Docker access and the image entry point.
EOF
}

fail() {
    echo "E1 ERROR: $*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --check-only)
            CHECK_ONLY=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            fail "unknown argument: $1"
            ;;
    esac
    shift
done

if [[ "$(uname -m)" != "x86_64" ]]; then
    fail "E1 requires an x86-64 Linux host"
fi

if ! command -v docker >/dev/null 2>&1; then
    fail "Docker is not installed or is not on PATH"
fi

declare -a DOCKER
if docker version >/dev/null 2>&1; then
    DOCKER=(docker)
    echo "E1 Docker access: direct"
elif command -v sudo >/dev/null 2>&1 && sudo -n docker version >/dev/null 2>&1; then
    DOCKER=(sudo -n docker)
    echo "E1 Docker access: passwordless sudo"
elif [[ -t 0 ]] && command -v sudo >/dev/null 2>&1; then
    echo "Direct Docker access is unavailable; requesting sudo authorization."
    if sudo docker version >/dev/null; then
        DOCKER=(sudo docker)
        echo "E1 Docker access: interactive sudo"
    else
        fail "sudo could not access the Docker daemon"
    fi
else
    fail "cannot access the Docker daemon and noninteractive sudo is unavailable. Configure Docker access or run this script from an interactive terminal with sudo permission"
fi

echo "E1 image: $IMAGE"
"${DOCKER[@]}" pull "$IMAGE"

if $CHECK_ONLY; then
    "${DOCKER[@]}" run --rm "$IMAGE" bash -lc \
        'test -f /opt/icflownet-e1/run_e1.sh'
    echo "E1 container PASS"
    exit 0
fi

command -v tar >/dev/null 2>&1 || fail "tar is required to collect E1 evidence"

"${DOCKER[@]}" rm -f "$CONTAINER" >/dev/null 2>&1 || true

if ! "${DOCKER[@]}" run --name "$CONTAINER" --init \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    "$IMAGE" \
    /opt/icflownet-e1/run_e1.sh \
    --sample zydis \
    --output /output/zydis; then
    echo "E1 ERROR: container experiment failed; container retained as $CONTAINER" >&2
    exit 1
fi

mkdir -p "$(dirname -- "$RESULT_ROOT")"
if [[ -e "$RESULT_ROOT" ]]; then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    previous="${RESULT_ROOT}.previous.${timestamp}"
    if [[ -e "$previous" ]]; then
        previous="${previous}.$$"
    fi
    mv -- "$RESULT_ROOT" "$previous"
    echo "Previous E1 output moved to: $previous"
fi
mkdir -p "$RESULT_ROOT"

if ! "${DOCKER[@]}" cp "$CONTAINER:/output/zydis/." - | tar -xf - -C "$RESULT_ROOT"; then
    echo "E1 ERROR: could not copy evidence; container retained as $CONTAINER" >&2
    exit 1
fi

"${DOCKER[@]}" rm "$CONTAINER" >/dev/null

REPORT="$RESULT_ROOT/verification.json"
[[ -f "$REPORT" ]] || fail "verification report was not copied to $REPORT"
grep -Eq '"status"[[:space:]]*:[[:space:]]*"PASS"' "$REPORT" || \
    fail "verification report does not contain PASS"

echo "E1 evidence: $RESULT_ROOT"
echo "E1 host workflow PASS"
