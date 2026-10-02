#!/usr/bin/env bash
# Manejo de la VM spot con 2×H100 en Azure, desde la Mac.
#
#   scripts/vm.sh status     estado y IP
#   scripts/vm.sh start      encender (empieza a cobrar, ~$2.58/h)
#   scripts/vm.sh stop       apagar y liberar (deja de cobrar; el disco se conserva)
#   scripts/vm.sh ssh        abrir una terminal en la VM
#   scripts/vm.sh run CMD    correr un comando en la VM
#   scripts/vm.sh sync       copiar este repo a la VM (rsync, sin .git ni datos)
#   scripts/vm.sh fetch      traer results/ de la VM a la Mac
#   scripts/vm.sh drivers    instalar drivers NVIDIA via extensión de Azure (solo si falta nvidia-smi)
#
# Precondición: `az login` hecho en la Mac y la clave SSH de la Mac autorizada en la VM.
# Los identificadores de Azure (grupo de recursos, nombre de la VM, usuario) viven en
# scripts/vm.env, que NO se versiona (el repo es público). Copiar vm.env.example.
set -euo pipefail

ENV_FILE="$(dirname "$0")/vm.env"
[ -f "$ENV_FILE" ] || { echo "Falta $ENV_FILE. Copiá scripts/vm.env.example a scripts/vm.env y completalo."; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"
: "${RG:?falta RG en vm.env}" "${VM:?falta VM en vm.env}" "${USER:?falta USER en vm.env}"
REMOTE_DIR="${REMOTE_DIR:-~/rnn-lookup}"

ip() { az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv; }

case "${1:-status}" in
  status)
    az vm show -d -g "$RG" -n "$VM" --query "{estado:powerState, ip:publicIps, tamaño:hardwareProfile.vmSize}" -o table
    ;;
  start)
    az vm start -g "$RG" -n "$VM" -o none
    echo "Encendida. IP: $(ip)  (recordá apagarla con: scripts/vm.sh stop)"
    ;;
  stop)
    az vm deallocate -g "$RG" -n "$VM" -o none
    echo "Apagada y liberada (no cobra)."
    ;;
  ssh)
    ssh -o StrictHostKeyChecking=accept-new "$USER@$(ip)"
    ;;
  run)
    shift
    ssh -o StrictHostKeyChecking=accept-new "$USER@$(ip)" "$@"
    ;;
  sync)
    rsync -az --delete \
      --exclude '.git' --exclude '/.venv' --exclude '/results/' --exclude '/data/' --exclude '__pycache__' \
      -e "ssh -o StrictHostKeyChecking=accept-new" \
      ./ "$USER@$(ip):$REMOTE_DIR/"
    echo "Repo copiado a $VM:$REMOTE_DIR"
    ;;
  fetch)
    mkdir -p results
    rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" \
      "$USER@$(ip):$REMOTE_DIR/results/" ./results/
    echo "results/ actualizado desde la VM"
    ;;
  drivers)
    az vm extension set -g "$RG" --vm-name "$VM" \
      --name NvidiaGpuDriverLinux --publisher Microsoft.HPC -o none
    echo "Extensión de drivers instalada. Reiniciá la VM si nvidia-smi sigue sin aparecer."
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1
    ;;
esac
