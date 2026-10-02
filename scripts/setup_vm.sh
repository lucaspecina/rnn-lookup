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

# Almacenamiento. El disco del sistema de la VM está casi lleno (otros proyectos).
# El disco temporal de Azure (/mnt, 256 GB) es efímero: se vacía si la VM se
# desaloja o se reinicia. Alcanza para el entorno y el cache de datos, que este
# script regenera en minutos. Los resultados van a results/, en el disco persistente.
SCRATCH="${SCRATCH:-/mnt/rnn-lookup}"
sudo mkdir -p "$SCRATCH" && sudo chown "$(id -u):$(id -g)" "$SCRATCH"
export UV_CACHE_DIR="$SCRATCH/uv-cache"
export UV_PYTHON_INSTALL_DIR="$SCRATCH/uv-python"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

VENV="$SCRATCH/venv"
[ -x "$VENV/bin/python" ] || uv venv --python 3.11 "$VENV"
ln -sfn "$VENV" .venv
mkdir -p "$SCRATCH/data" results && ln -sfn "$SCRATCH/data" data
source .venv/bin/activate

# torch con CUDA 12.8 (H100 + driver reciente). Si el driver es viejo, cambiar a cu126.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # torchvision: zoology/model.py lo importa
# kernels Triton de Gated DeltaNet y familia (pide torch>=2.7, triton>=3.3)
uv pip install flash-linear-attention
# El Triton que trae torch (3.6) tiene un bug en Hopper en la pasada hacia atrás
# de Gated DeltaNet (fla issue #640): fla exige >=3.7.1 o tilelang.
uv pip install "triton>=3.7.1"
# Zoology (nuestro fork) sin sus deps automáticas: causal_conv1d y mamba_ssm son
# problemáticos y no hacen falta (fla usa su backend Triton para la conv corta).
# --no-build-isolation: el setup.py de Zoology importa torch al instalar
uv pip install --no-build-isolation --no-deps -e third_party/zoology
uv pip install numpy einops tqdm click pydantic pandas pyyaml rich ray \
               rotary-embedding-torch einx transformers wandb matplotlib seaborn tabulate
uv pip install -e .

python - <<'PY'
import torch, fla, zoology, rnn_lookup
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
print("fla", fla.__version__)
PY
echo "Listo. Prueba de humo:  source .venv/bin/activate && python -m zoology.launch experiments/mqar/smoke.py"
