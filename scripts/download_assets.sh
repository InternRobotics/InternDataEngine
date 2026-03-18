#!/bin/bash
# Download assets from HuggingFace.
# Usage:
#   bash scripts/download_assets.sh [OPTIONS]
#
# Options:
#   --min           Download only required scene assets (for quick testing)
#   --full          Download all scene assets including all robots and tasks (default)
#   --with-curobo   Also download CuRobo package
#   --with-drake    Also download panda_drake package
#   --local-dir DIR Where to save (default: current directory)
#
# Examples:
#   bash scripts/download_assets.sh --min
#   bash scripts/download_assets.sh --full --with-curobo --with-drake
#   bash scripts/download_assets.sh --min --with-curobo --local-dir /data/assets

set -e

REPO_ID="InternRobotics/InternData-A1"
REPO_TYPE="dataset"
ASSET_PREFIX="InternDataAssets"

MODE="full"
LOCAL_DIR="."
WITH_CUROBO=false
WITH_DRAKE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --min)          MODE="min";       shift ;;
        --full)         MODE="full";      shift ;;
        --with-curobo)  WITH_CUROBO=true; shift ;;
        --with-drake)   WITH_DRAKE=true;  shift ;;
        --local-dir)    LOCAL_DIR="$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

info() { echo -e "\033[32m[INFO]\033[0m $*"; }

download() {
    info "Downloading $2 ..."
    hf download "$REPO_ID" --repo-type "$REPO_TYPE" --include "$1" --local-dir "$LOCAL_DIR"
}

# --- Scene assets: required (both modes) ---
info "========== Downloading required scene assets =========="
REQUIRED_DIRS=("background_textures" "envmap_lib" "floor_textures" "table_textures" "table0")
for dir in "${REQUIRED_DIRS[@]}"; do
    download "${ASSET_PREFIX}/assets/${dir}/*" "$dir"
done
download "${ASSET_PREFIX}/assets/table_info.json" "table_info.json"

# --- Scene assets: full only (all robots + all tasks) ---
if [[ "$MODE" == "full" ]]; then
    info "========== Downloading all robots and tasks =========="
    for robot in lift2 franka frankarobotiq split_aloha_mid_360 G1_120s; do
        download "${ASSET_PREFIX}/assets/${robot}/*" "robot: ${robot}"
    done
    for task in basic art long_horizon pick_and_place; do
        download "${ASSET_PREFIX}/assets/${task}/*" "task: ${task}"
    done
fi

# --- CuRobo ---
if [[ "$WITH_CUROBO" == true ]]; then
    info "========== Downloading CuRobo =========="
    download "${ASSET_PREFIX}/curobo/*" "curobo"
fi

# --- panda_drake ---
if [[ "$WITH_DRAKE" == true ]]; then
    info "========== Downloading panda_drake =========="
    download "${ASSET_PREFIX}/panda_drake/*" "panda_drake"
fi

info "Done! (mode=${MODE}, curobo=${WITH_CUROBO}, drake=${WITH_DRAKE}, local-dir=${LOCAL_DIR})"
