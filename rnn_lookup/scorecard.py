"""Tarjeta de puntajes: resume una carpeta de resultados en una tabla por juego.

Lee results/<banco>/<juego>/<corrida>/{config.json, model.json, metrics.jsonl}
(lo que escribe nuestro fork de Zoology) y arma, por juego, una fila por
arquitectura con:
  - la mejor precisión en cada "slice" (por ejemplo, por cantidad de pares),
    tomando el máximo sobre épocas, como hace Zoology, y el mejor learning rate
  - memoria del estado (bytes que carga el modelo al generar), parámetros,
    épocas corridas y minutos de entrenamiento

Uso:
    python -m rnn_lookup.scorecard results/bench_v1
Escribe results/<banco>/scorecard.md y scorecard.csv.
"""
import json
import sys
from pathlib import Path

import pandas as pd


def load_run(run_dir: Path) -> dict | None:
    cfg_p, met_p = run_dir / "config.json", run_dir / "metrics.jsonl"
    if not cfg_p.exists() or not met_p.exists():
        return None
    cfg = json.loads(cfg_p.read_text())
    model_info = json.loads((run_dir / "model.json").read_text()) if (run_dir / "model.json").exists() else {}
    rows = [json.loads(l) for l in met_p.read_text().splitlines() if l.strip()]
    valid = [r for r in rows if "valid/accuracy" in r]
    if not valid:
        return None
    best = {k: max(r[k] for r in valid if k in r) for r in valid for k in r if k.startswith("valid/")}
    m = cfg["model"]
    return {
        "arch": f'{m["name"]}-d{m["d_model"]}',
        "lr": cfg["learning_rate"],
        "epochs": len(valid),
        "minutes": round(rows[-1].get("t", 0) / 60, 1),
        "params": model_info.get("num_parameters"),
        "state_bytes": model_info.get("state_size"),
        **best,
    }


def scorecard(bank_dir: Path) -> pd.DataFrame:
    records = []
    for game_dir in sorted(p for p in bank_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in game_dir.iterdir() if p.is_dir()):
            r = load_run(run_dir)
            if r is not None:
                records.append({"game": game_dir.name, **r})
    if not records:
        raise SystemExit(f"No hay corridas con resultados en {bank_dir}")
    df = pd.DataFrame(records)
    # mejor learning rate por (juego, arquitectura), según la precisión global
    idx = df.groupby(["game", "arch"])["valid/accuracy"].idxmax()
    return df.loc[idx].sort_values(["game", "arch"]).reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    out = []
    for game, g in df.groupby("game", sort=True):
        slice_cols = sorted(
            [c for c in g.columns if c.startswith("valid/") and c != "valid/accuracy" and c != "valid/loss" and g[c].notna().any()],
            key=lambda c: (c.rsplit("-", 1)[0], float(c.rsplit("-", 1)[1])),
        )
        cols = ["arch", "valid/accuracy"] + slice_cols + ["state_bytes", "params", "lr", "epochs", "minutes"]
        t = g[cols].copy()
        t.columns = [c.replace("valid/", "").replace("/accuracy-", "=") for c in cols]
        for c in t.columns:
            if t[c].dtype.kind == "f" and c not in ("lr", "minutes"):
                t[c] = t[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
        out.append(f"## {game}\n\n" + t.to_markdown(index=False) + "\n")
    return "\n".join(out)


if __name__ == "__main__":
    bank = Path(sys.argv[1] if len(sys.argv) > 1 else "results/bench_v1")
    df = scorecard(bank)
    (bank / "scorecard.csv").write_text(df.to_csv(index=False))
    md = to_markdown(df)
    (bank / "scorecard.md").write_text(md)
    print(md)
