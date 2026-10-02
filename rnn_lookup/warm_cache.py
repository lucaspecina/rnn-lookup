"""Genera y cachea los datos de un experimento antes de lanzarlo en paralelo.

Zoology genera los datos la primera vez que los necesita y los guarda en
`data.cache_dir`. Si varias corridas arrancan a la vez con los mismos datos,
compiten por escribir el mismo archivo. Correr esto primero evita la carrera.

    python -m rnn_lookup.warm_cache experiments/bench_v1.py
"""
import importlib.util
import sys
import time

from zoology.data.utils import prepare_data


def main(path: str):
    spec = importlib.util.spec_from_file_location("exp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = {}
    for cfg in mod.configs:
        # serialize_as_any: si no, pydantic serializa cada segmento como DataSegmentConfig
        # (la clase declarada) y pierde num_kv_pairs, name, etc.
        key = cfg.data.model_dump_json(serialize_as_any=True)
        if key not in seen:
            seen[key] = cfg
    print(f"{len(mod.configs)} corridas, {len(seen)} configuraciones de datos distintas")
    for i, cfg in enumerate(seen.values(), 1):
        t = time.time()
        prepare_data(cfg.data)
        print(f"[{i}/{len(seen)}] {cfg.sweep_id}: listo en {time.time() - t:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
