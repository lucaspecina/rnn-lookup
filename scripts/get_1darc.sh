#!/usr/bin/env bash
# Baja el dataset 1D-ARC (Xu et al. 2023, MIT) a data/1D-ARC para usarlo como examen.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ ! -d data/1D-ARC ]; then
  git clone -q --depth 1 https://github.com/khalil-research/1D-ARC.git data/1D-ARC
fi
echo "1D-ARC: $(ls data/1D-ARC/dataset | wc -l) tipos de tarea, $(find data/1D-ARC/dataset -name '*.json' | wc -l) instancias"
