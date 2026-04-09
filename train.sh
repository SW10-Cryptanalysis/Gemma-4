#!/bin/bash
set -e

# 1. Navigate to your mounted workspace
cd /work

# 2. Clone the repository and specific branch if it doesn't exist yet
if [ ! -d "Gemma" ]; then
    echo "Cloning repository..."
    git clone -b main https://github.com/SW10-Cryptanalysis/Gemma-4.git
    
fi

cd Gemma-4
mkdir -p logs
export HF_HUB_ENABLE_HF_TRANSFER=1

# 3. Dynamically count available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Training Job started on $(hostname) at $(date) with $NUM_GPUS GPU(s)"

# Safely set CUDA_VISIBLE_DEVICES based on the detected count
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
nvidia-smi

# Ensure uv is installed 
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

export OMP_NUM_THREADS=16  # Increased for H100's stronger CPUs to feed the dataloader

# 4. H100 NVLink & NCCL Optimizations
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [ "$NUM_GPUS" -gt 1 ]; then
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
fi

echo "Creating virtual environment..."
uv venv
source .venv/bin/activate

# Install project dependencies (removed --system)
uv pip install -e .

# Install hf_transfer to enable faster Hugging Face downloads
uv pip install hf_transfer

# 5. Flash Attention 2 Installation
echo "Installing Flash Attention 2..."
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu130torch2.10-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl

MASTER_PORT=$((10000 + $RANDOM % 20000))

# 6. Launch Training
echo "Launching torchrun with $NUM_GPUS processes..."
uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m src.train

echo "Training Job finished at $(date)"