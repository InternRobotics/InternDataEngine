#!/usr/bin/env bash
set -ex

cd YOUR_PATH/openpi

export USE_TF=0
export USE_TORCH=0
export USE_JAX=1
export IMAGEIO_FFMPEG_EXE=ffmpeg
# JAX GPU memory fraction
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# ============================================================================
# NCCL Configuration
# ============================================================================
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ============================================================================
# Platform-Injected Configuration
# ============================================================================
# The platform automatically injects these when DISTRIBUTED_JOB=true:
#   - NCCL_IB_HCA, NCCL_IB_GID_INDEX, NCCL_SOCKET_IFNAME
#   - NODE_RANK, NODE_COUNT, MASTER_ADDR, PROC_PER_NODE
#   - CUDA_VISIBLE_DEVICES
# We trust and use these platform configurations directly.
# ============================================================================

echo ""
echo "=========================================="
echo "Platform Configuration"
echo "=========================================="
echo "NODE_RANK:             ${NODE_RANK:-<not set>}"
echo "NODE_COUNT:            ${NODE_COUNT:-<not set>}"
echo "MASTER_ADDR:           ${MASTER_ADDR:-<not set>}"
echo "NCCL_IB_HCA:           ${NCCL_IB_HCA:-<not set>}"
echo "NCCL_IB_GID_INDEX:     ${NCCL_IB_GID_INDEX:-<not set>}"
echo "NCCL_SOCKET_IFNAME:    ${NCCL_SOCKET_IFNAME:-<not set>}"
echo "=========================================="
echo ""

# ============================================================================
# NCCL Transport Configuration
# ============================================================================
# Use platform-injected configuration if available, otherwise fallback
# ============================================================================

if [ -n "${NCCL_IB_HCA:-}" ]; then
  # Platform has configured InfiniBand
  echo "[NCCL] ✓ Using platform-injected InfiniBand configuration"
  
  # Only set NCCL_NET if not already set
  if [ -z "${NCCL_NET:-}" ]; then
    export NCCL_NET="IB"
  fi
  
  # Set IB timeout if not already set
  if [ -z "${NCCL_IB_TIMEOUT:-}" ]; then
    export NCCL_IB_TIMEOUT=23
  fi
  
  echo "[NCCL]   NCCL_NET:          ${NCCL_NET}"
  echo "[NCCL]   NCCL_IB_HCA:       ${NCCL_IB_HCA}"
  echo "[NCCL]   NCCL_IB_GID_INDEX: ${NCCL_IB_GID_INDEX}"
  echo "[NCCL]   NCCL_IB_TIMEOUT:   ${NCCL_IB_TIMEOUT}"
  
elif [ -n "${NCCL_SOCKET_IFNAME:-}" ]; then
  # Platform has configured Socket
  echo "[NCCL] ✓ Using platform-injected Socket configuration"
  
  if [ -z "${NCCL_NET:-}" ]; then
    export NCCL_NET="Socket"
  fi
  
  echo "[NCCL]   NCCL_NET:           ${NCCL_NET}"
  echo "[NCCL]   NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
  
else
  # No platform injection - use OPENPI_NCCL_NET preference
  echo "[NCCL] ⚠️  No platform-injected NCCL configuration"
  
  if [ "${OPENPI_NCCL_NET:-IB}" = "IB" ]; then
    echo "[NCCL] ✗ InfiniBand requested but not configured by platform"
    echo "[NCCL] ✗ Falling back to Socket transport"
    export NCCL_NET="Socket"
    export NCCL_IB_DISABLE=1
  else
    export NCCL_NET="Socket"
    export NCCL_IB_DISABLE=1
    echo "[NCCL] Using Socket transport"
  fi
fi

echo ""

# ============================================================================
# JAX Distributed Configuration
# ============================================================================
# Map platform variables to JAX variables
# ============================================================================

echo "=========================================="
echo "JAX Distributed Configuration"
echo "=========================================="

JAX_COORDINATOR_PORT="${JAX_COORDINATOR_PORT:-12345}"

# Set JAX coordinator address
if [ -z "${JAX_COORDINATOR_ADDRESS:-}" ] && [ -n "${MASTER_ADDR:-}" ]; then
  export JAX_COORDINATOR_ADDRESS="${MASTER_ADDR}:${JAX_COORDINATOR_PORT}"
  echo "[JAX] ✓ Coordinator: ${JAX_COORDINATOR_ADDRESS} (from MASTER_ADDR)"
elif [ -n "${JAX_COORDINATOR_ADDRESS:-}" ]; then
  echo "[JAX] ✓ Coordinator: ${JAX_COORDINATOR_ADDRESS}"
else
  echo "[JAX] ✗ WARNING: No coordinator address set!"
fi

# Set JAX process count
if [ -z "${JAX_PROCESS_COUNT:-}" ] && [ -n "${NODE_COUNT:-}" ]; then
  export JAX_PROCESS_COUNT="${NODE_COUNT}"
  echo "[JAX] ✓ Process count: ${JAX_PROCESS_COUNT} (from NODE_COUNT)"
elif [ -n "${JAX_PROCESS_COUNT:-}" ]; then
  echo "[JAX] ✓ Process count: ${JAX_PROCESS_COUNT}"
fi

# Set JAX process index
if [ -z "${JAX_PROCESS_INDEX:-}" ] && [ -n "${NODE_RANK:-}" ]; then
  export JAX_PROCESS_INDEX="${NODE_RANK}"
  echo "[JAX] ✓ Process index: ${JAX_PROCESS_INDEX} (from NODE_RANK)"
elif [ -n "${JAX_PROCESS_INDEX:-}" ]; then
  echo "[JAX] ✓ Process index: ${JAX_PROCESS_INDEX}"
fi

echo "=========================================="
echo ""

# ============================================================================
# Python Environment
# ============================================================================
export PYTHONPATH=YOUR_PATH/openpi/src:YOUR_PATH/openpi/packages/openpi-client/src:YOUR_PATH/openpi/third_party/lerobot:${PYTHONPATH}
conda activate pi0

# ============================================================================
# Configuration Summary
# ============================================================================

echo "=========================================="
echo "Configuration Summary"
echo "=========================================="
echo "NCCL_NET:              ${NCCL_NET:-<not set>}"
echo "NCCL_IB_HCA:           ${NCCL_IB_HCA:-<not set>}"
echo "NCCL_IB_GID_INDEX:     ${NCCL_IB_GID_INDEX:-<not set>}"
echo "NCCL_SOCKET_IFNAME:    ${NCCL_SOCKET_IFNAME:-<not set>}"
echo "JAX_COORDINATOR:       ${JAX_COORDINATOR_ADDRESS:-<not set>}"
echo "JAX_PROCESS_COUNT:     ${JAX_PROCESS_COUNT:-<not set>}"
echo "JAX_PROCESS_INDEX:     ${JAX_PROCESS_INDEX:-<not set>}"
echo "=========================================="
echo ""

# ============================================================================
# Display Host Information
# ============================================================================

python - <<'EOF'
import socket
import os
import jax
hostname = socket.gethostname()
devices = jax.local_devices()
device_count = len(devices)
device_ids = [d.id for d in devices]
print(f"[JAX] host={hostname}, devices={device_count}xgpu, ids={device_ids}")
print(f"[JAX] JAX_COORDINATOR_ADDRESS={os.environ.get('JAX_COORDINATOR_ADDRESS', '<not set>')}")
print(f"[JAX] JAX_PROCESS_COUNT={os.environ.get('JAX_PROCESS_COUNT', '<not set>')}")
print(f"[JAX] JAX_PROCESS_INDEX={os.environ.get('JAX_PROCESS_INDEX', '<not set>')}")
EOF

# ============================================================================
# Launch Training
# ============================================================================

# Determine experiment name based on transport
if [ "${OPENPI_DEBUG_SINGLE_GPU:-0}" = "1" ]; then
  EXP_NAME="${EXP_NAME:-dev_jax_single_gpu}"
  echo "[DEBUG] Running in single-GPU mode"
else
  EXP_NAME="${EXP_NAME:-dev_jax_multinode_ib}"
fi

echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo "Experiment: $EXP_NAME"
echo "=========================================="
echo ""

ulimit -n 1000000

python scripts/train_jax_multinode.py \
    pretrain-interndata-a1 \
  --exp-name=pretrain-interndata-a1 \
  --num_workers=12 \
  --batch_size=512 \
  --num_train_steps=2000000 \
  --save_interval=5000
