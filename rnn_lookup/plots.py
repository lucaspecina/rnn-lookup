"""Gráficos del banco: una figura por juego y la curva de compromiso memoria vs. precisión.

    python -m rnn_lookup.plots results/bench_v1
Escribe results/<banco>/plots/<juego>.png y results/<banco>/plots/mqar_tradeoff.png.

Convenciones (del skill dataviz): color por familia de arquitectura, en orden
fijo; marcador por tamaño; líneas finas; grilla y ejes recesivos; texto en tinta
neutra, nunca en el color de la serie.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rnn_lookup.scorecard import scorecard

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
FAMILY_COLOR = {"attention": "#2a78d6", "sliding_window": "#eb6834", "gated_delta_net": "#1baf7a"}
MARKERS = ["o", "s", "^", "D", "v"]
X_AXIS = {  # qué perilla de dificultad va en el eje x de cada juego
    "mqar": "num_kv_pairs", "forgetting_mqar": "num_kv_pairs", "compositional_mqar": "num_kv_pairs",
    "cumulative_parity": "input_seq_len", "cumulative_majority": "input_seq_len",
}
X_LABEL = {"num_kv_pairs": "pares clave→valor por secuencia", "input_seq_len": "largo de la secuencia (tokens)"}
TITLE = {
    "mqar": "Recuerdo exacto (MQAR)", "forgetting_mqar": "Recuerdo con actualización",
    "compositional_mqar": "Recuerdo con claves compuestas", "cumulative_parity": "Paridad acumulada",
    "cumulative_majority": "Mayoría acumulada",
}


def family(arch: str) -> str:
    for f in ("sliding_window", "gated_delta_net", "attention"):
        if arch.startswith(f):
            return f
    return arch


def style_for(arch: str, seen: dict) -> dict:
    """Color por familia, marcador por variante dentro de la familia, punteado para la variante 'neg'."""
    fam = family(arch)
    idx = seen.setdefault(fam, [])
    if arch not in idx:
        idx.append(arch)
    return dict(
        color=FAMILY_COLOR[fam],
        marker=MARKERS[idx.index(arch) % len(MARKERS)],
        linestyle="--" if "_neg" in arch else "-",
    )


def _axes(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel("precisión en las preguntas", color=INK2, fontsize=10)
    ax.set_ylim(-0.02, 1.05)


def plot_game(df: pd.DataFrame, game: str, out: Path):
    g = df[df.game == game]
    xkey = X_AXIS[game]
    # solo las dificultades que este juego realmente tiene (el DataFrame junta columnas de todos los juegos)
    cols = sorted([c for c in g.columns if c.startswith(f"valid/{xkey}/accuracy-") and g[c].notna().any()],
                  key=lambda c: float(c.rsplit("-", 1)[1]))
    xs = [float(c.rsplit("-", 1)[1]) for c in cols]
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    seen = {}
    for _, row in g.sort_values("arch").iterrows():
        st = style_for(row["arch"], seen)
        ys = [row[c] for c in cols]
        ax.plot(xs, ys, linewidth=1.5, markersize=6, label=row["arch"], **st)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs])
    _axes(ax, TITLE.get(game, game), X_LABEL[xkey])
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out / f"{game}.png", dpi=160)
    plt.close(fig)


def plot_tradeoff(df: pd.DataFrame, out: Path, game: str = "mqar"):
    """Curva de compromiso al estilo Based: memoria del estado (bytes) vs. precisión."""
    g = df[df.game == game]
    if g.empty:
        return
    hard = max((c for c in g.columns if c.startswith("valid/num_kv_pairs/accuracy-") and g[c].notna().any()),
               key=lambda c: float(c.rsplit("-", 1)[1]))
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    seen = {}
    for _, row in g.sort_values("arch").iterrows():
        st = style_for(row["arch"], seen)
        ax.scatter(row["state_bytes"], row[hard], s=70, color=st["color"], marker=st["marker"], edgecolors=SURFACE, linewidths=1.5, zorder=3, label=row["arch"])
    ax.set_xscale("log")
    _axes(ax, f"Memoria del estado vs. precisión con {hard.rsplit('-', 1)[1]} pares", "bytes de estado que carga el modelo al generar (log)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out / f"{game}_tradeoff.png", dpi=160)
    plt.close(fig)


def main(bank: Path):
    df = scorecard(bank)
    out = bank / "plots"
    out.mkdir(exist_ok=True)
    for game in sorted(df.game.unique()):
        if game in X_AXIS:
            plot_game(df, game, out)
    plot_tradeoff(df, out)
    print("gráficos en", out, ":", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/bench_v1"))
