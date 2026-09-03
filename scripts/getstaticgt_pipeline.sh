#!/bin/bash

# Build static ground-truth artifacts for one dataset environment by chaining
# the parsing, angr CFG generation, and result-merging steps.

# Get environment name from argument
env="$1"
if [ -z "$env" ]; then
    echo "Usage: $0 <env>"
    exit 1
fi

# Define directory paths. Set DATA_WORK_ROOT to the parent directory that holds
# env folders such as total/, env_0/, and dynamic_static/.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_WORK_ROOT="${DATA_WORK_ROOT:-${WORK_ROOT:-}}"
if [ -z "$DATA_WORK_ROOT" ]; then
    echo "Set DATA_WORK_ROOT or WORK_ROOT to the artifact work directory."
    exit 1
fi

BASE_DIR="${BASE_DIR:-$DATA_WORK_ROOT/${env}}"
BINARY_DIR="$BASE_DIR/binary"
ANGRINFO_DIR="$BASE_DIR/angrinfo"
SOURCEINFO_DIR="$BASE_DIR/sourceinfo"
RES_DIR="$BASE_DIR/res"
MAP_FILE="$BASE_DIR/gttobin.json"
SCRIPTS_DIR="${STATIC_GT_SCRIPTS_DIR:-$REPO_ROOT/src/groundtruth/static}"


# Create necessary directories
mkdir -p "$BASE_DIR" "$ANGRINFO_DIR" "$SOURCEINFO_DIR" "$RES_DIR"

# Find all ELF binaries in a directory
find_elf_binaries() {
    local parent_dir=$1
    find "$parent_dir" -type f -exec file {} \; | grep -i "ELF" | cut -d':' -f1
}

# Run a Python script and handle errors
run_script() {
    script_path=$1
    shift
    args="$@"

    echo "========================================"
    echo "Running: $script_path"
    echo "Arguments: $args"
    echo "========================================"

    python3 "$script_path" $args
    if [ $? -ne 0 ]; then
        echo "❌ Error: $script_path failed to execute properly."
        return 1
    else
        echo "✅ $script_path completed successfully."
        echo ""
        return 0
    fi
}

# Process a single binary
process_binary() {
    binary_path=$1
    binary_index=$2
    binary_name=$(basename "$binary_path")

    echo "========================================="
    echo "Processing binary: $binary_name"
    echo "========================================="

    binary_angrinfo_dir="$ANGRINFO_DIR/$binary_name"
    binary_sourceinfo_dir="$SOURCEINFO_DIR/$binary_name"
    binary_res_dir="$RES_DIR/$binary_name"

    if [ -d "$binary_res_dir" ] && [ "$(ls -A "$binary_res_dir" 2>/dev/null)" ]; then
        echo "Results directory for $binary_name already exists. Skipping processing."
        return 0
    fi

    mkdir -p "$binary_angrinfo_dir" "$binary_sourceinfo_dir" "$binary_res_dir"

#    run_script "$SCRIPTS_DIR/parsebylabel.py" "$binary_path" "$binary_sourceinfo_dir" || {
        #echo "❌ parsebylabel.py failed for $binary_name, skipping to next binary"
        #return 0
    #}

    if ! python3 "$SCRIPTS_DIR/parsebylabel.py" "$binary_path" "$binary_sourceinfo_dir"; then
        echo "❌ parsebylabel.py failed for $binary_name, skipping to next binary"
        return 2
    else
        echo "✅ parsebylabel.py completed successfully."
    fi
    
    run_script "$SCRIPTS_DIR/angrcfginfo.py" "$binary_path" "$binary_angrinfo_dir" "$binary_sourceinfo_dir" "$binary_res_dir" "$BASE_DIR" "$binary_index" || return 1
    run_script "$SCRIPTS_DIR/processdcallreturn.py" "$binary_name" "$binary_angrinfo_dir" "$binary_res_dir"  || return 1
    run_script "$SCRIPTS_DIR/processicallreturn.py" "$binary_name" "$binary_angrinfo_dir" "$binary_sourceinfo_dir" "$binary_res_dir"  || return 1
    run_script "$SCRIPTS_DIR/getfuncascalleetoicallins.py" "$binary_name" "$binary_sourceinfo_dir" || return 1
    run_script "$SCRIPTS_DIR/combinefuncascalleetocallins.py" "$binary_name" "$binary_angrinfo_dir" "$binary_sourceinfo_dir"  || return 1
    run_script "$SCRIPTS_DIR/gettcfunctocallee.py" "$binary_name" "$binary_angrinfo_dir" "$binary_sourceinfo_dir"  || return 1
    run_script "$SCRIPTS_DIR/gettcfuncallpathret.py" "$binary_name" "$binary_angrinfo_dir" "$binary_sourceinfo_dir" || return 1
    run_script "$SCRIPTS_DIR/getrettoaftercallfortc.py" "$binary_name" "$binary_angrinfo_dir" "$binary_res_dir" || return 1
    run_script "$SCRIPTS_DIR/mergeret.py" "$binary_name" "$binary_res_dir" "$binary_path" "$MAP_FILE" || return 1
    mv "$binary_angrinfo_dir"/*_nodelookup.json "$binary_res_dir"/
    
    #delete binary_angr_dir and binary_source_dir to save storage
    echo "Cleaning up temporary directories for $binary_name"
    #rm -rf "$binary_angrinfo_dir"
    #rm -rf "$binary_sourceinfo_dir"   
    echo "✅ Processing of $binary_name completed successfully"
    return 0
}

# Main execution
echo "🔧 Starting batch processing of binaries..."

if [ -z "$BINARY_DIR" ]; then
    echo "❌ Error: BINARY_DIR is not set."
    exit 1
fi

# Create map file if not exist
if [ ! -f "$MAP_FILE" ]; then
    mkdir -p "$(dirname "$MAP_FILE")"
    touch "$MAP_FILE"
    echo "🆕 Created map file: $MAP_FILE"
else
    echo "📄 Found map file: $MAP_FILE"
fi

# Initialize counters
total_count=0
success_count=0
failure_count=0
skipped_count=0

binary_files=$(find_elf_binaries "$BINARY_DIR")
if [ -z "$binary_files" ]; then
    echo "⚠️ No ELF binaries found in $BINARY_DIR"
    exit 1
fi

INDEX_FILE="$BASE_DIR/indextores.json"
if [ -f "$INDEX_FILE" ]; then
    last_index=$(jq 'keys | map(tonumber) | max' "$INDEX_FILE" 2>/dev/null)
    if [ -n "$last_index" ]; then
        binary_index=$((last_index + 1))
    else
        binary_index=0
    fi
else
    binary_index=0
fi

for binary in $binary_files; do
    total_count=$((total_count + 1))
    binary_name=$(basename "$binary")
    binary_res_dir="$RES_DIR/$binary_name"

    echo "🚀 Processing binary: $binary_name"

    if [ -d "$binary_res_dir" ] && [ "$(ls -A "$binary_res_dir" 2>/dev/null)" ]; then
        echo "⏩ Skipping: $binary_name already processed"
        skipped_count=$((skipped_count + 1))
        continue
    fi

    if process_binary "$binary" "$binary_index"; then
        success_count=$((success_count + 1))
        binary_index=$((binary_index + 1))
    else
        failure_count=$((failure_count + 1))
        echo "❌ ERROR: Failed to process $binary_name"
    fi

    echo "📊 Progress: $success_count success, $failure_count failed, $skipped_count skipped out of $total_count"
done

run_script "$SCRIPTS_DIR/getindextobin.py" "$BASE_DIR" || exit 1

echo "✅ Batch processing complete!"
echo "📈 Final results: $success_count success, $failure_count failed, $skipped_count skipped out of $total_count"
