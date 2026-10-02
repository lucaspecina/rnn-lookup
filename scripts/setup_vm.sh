#!/usr/bin/env bash
# Prepara el entorno en la VM (Ubuntu 22.04, GPUs NVIDIA). Idempotente.
# Se corre EN la VM, desde ~/rnn-lookup (copiado con `scripts/vm.sh sync`):
#     bash scripts/setup_vm.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v nvidia-smi >/dev/null; then
  echo "No hay nvidia-smi. Desde la Mac: scripts/vm.sh drivers, y después reiniciar la VM."; exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

[ -d .venv ] || uv venv --python 3.11 .venv
source .venv/bin/activate

# torch con CUDA 12.8 (H100 + driver reciente). Si el driver es viejo, cambiar a cu126.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # torchvision: zoology/model.py lo importa
# kernels Triton de Gated DeltaNet y familia (pide torch>=2.7, triton>=3.3)
uv pip install flash-linear-attention
# Zoology (nuestro fork) sin sus deps automáticas: causal_conv1d y mamba_ssm son
# problemáticos y no hacen falta (fla usa su backend Triton para la conv corta).
# --no-build-isolation: el setup.py de Zoology importa torch al instalar
uv pip install --no-build-isolation --no-deps -e third_party/zoology
uv pip install numpy einops tqdm click pydantic pandas pyyaml rich ray \
               rotary-embedding-torch einx transformers wandb matplotlib seaborn
uv pip install -e .

mkdir -p data results
python - <<'PY'
import torch, fla, zoology, rnn_lookup
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
print("fla", fla.__version__)
PY
echo "Listo. Prueba de humo:  source .venv/bin/activate && python -m zoology.launch experiments/mqar/smoke.py"
