"""Tablero del banco: una página HTML que muestra los juegos, las arquitecturas,
el protocolo y los resultados, generada desde la config del experimento y la
carpeta de resultados.

    python -m rnn_lookup.dashboard experiments/bench_v1.py results/bench_v1
Escribe results/<banco>/tablero.html (autocontenido: SVG inline, sin librerías).
"""
import html
import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd

from rnn_lookup.scorecard import scorecard

FAMILY = {"attention": ("Atención", "#2a78d6"), "sliding_window": ("Ventana", "#eb6834"), "gated_delta_net": ("Gated DeltaNet", "#1baf7a")}
ROLE_LABEL = {"clave": "clave", "valor": "valor", "pregunta": "pregunta", "relleno": "relleno", "bit": "bit"}
GAME_TEXT = {
    "mqar": ("Recuerdo exacto (MQAR)", "Primero pares clave→valor, después preguntas sueltas. En cada pregunta hay que responder el valor que acompañaba a esa clave. Mide memoria exacta.", "num_kv_pairs"),
    "forgetting_mqar": ("Recuerdo con actualización", "Algunas claves aparecen dos veces con valores distintos. Vale el último. Mide si el modelo puede sobreescribir lo que guardó.", "num_kv_pairs"),
    "compositional_mqar": ("Recuerdo con claves compuestas", "La clave son dos tokens (K1, K2) y el valor depende de los dos juntos: K1 solo no alcanza. Mide composición.", "num_kv_pairs"),
    "cumulative_parity": ("Paridad acumulada", "Una tira de bits. En cada posición hay que decir si hasta ahí hubo una cantidad par o impar de unos. Mide seguimiento de estado; a los Transformers les cuesta.", "input_seq_len"),
    "cumulative_majority": ("Mayoría acumulada", "Una tira de bits. En cada posición hay que decir si hasta ahí hubo más unos que ceros. Mide conteo.", "input_seq_len"),
    "arc1d": ("Mini-ARC con objetos (1D-ARC)", "Filas de píxeles con fondo 0 y objetos (tramos de un color). Cada secuencia trae una regla del catálogo de 1D-ARC (mover, rellenar, vaciar, recolorear por tamaño, copiar patrón, estirar, quitar ruido...) con 3 ejemplos y una consulta. El examen incluye reglas nunca vistas y las 860 instancias reales del dataset 1D-ARC.", None),
    "rules": ("Reglas en contexto (mini-ARC)", "Cada secuencia trae una regla nueva: K ejemplos de entrada→salida separados por ';' y una consulta (naranja). Hay que inducir la regla de los ejemplos y producir la salida (la última tira verde). Las reglas son composiciones de primitivas (invertir, rotar, ordenar...). El examen incluye composiciones y primitivas nunca vistas.", None),
}


# ---------- juegos ----------

def example_roles(game: str, cfg):
    """Genera un ejemplo chico del juego y asigna un rol a cada token para pintarlo."""
    if game in ("rules", "arc1d"):
        if game == "rules":
            c = cfg.model_copy(update={"num_examples": 1, "num_demos": 2, "str_len": 4, "input_seq_len": 32})
            S = c.num_symbols
        else:
            c = cfg.model_copy(update={"num_examples": 1, "num_demos": 2, "width_min": 8, "width_max": 10, "input_seq_len": 64, "source": "gen", "rules": ["recolor_size"]})
            S = 10
        seg = c.build(seed=5)
        x, y = seg.inputs[0].tolist(), seg.labels[0].tolist()
        roles, side, last_q = [], "clave", False
        n_lab = sum(1 for t in y if t != -100)
        for i, t in enumerate(x):
            if t == S + 2:
                roles.append("relleno")           # PAD
            elif t == S:                          # →
                roles.append("relleno"); side = "valor"
            elif t == S + 1:                      # ;
                roles.append("relleno"); side = "clave"
            elif game == "arc1d" and t == 0:
                roles.append("relleno")          # fondo
            else:
                roles.append(side)
        # la última entrada (consulta) y su respuesta
        arrows = [i for i, t in enumerate(x) if t == S]
        q_start = ([i for i, t in enumerate(x) if t == S + 1][-1] + 1) if S + 1 in x else 0
        for i in range(q_start, arrows[-1]):
            roles[i] = "pregunta"
        display = [{S: "→", S + 1: ";", S + 2: "·"}.get(t, str(t)) for t in x]
        return x, [-100] * len(x), roles, display
    small = {"mqar": dict(vocab_size=64, input_seq_len=20, num_kv_pairs=3),
             "forgetting_mqar": dict(vocab_size=64, input_seq_len=24, num_kv_pairs=3, num_updates=2),
             "compositional_mqar": dict(vocab_size=64, input_seq_len=30, num_kv_pairs=4),
             "cumulative_parity": dict(vocab_size=16, input_seq_len=12),
             "cumulative_majority": dict(vocab_size=16, input_seq_len=12)}[game]
    c = cfg.model_copy(update={**small, "num_examples": 1})
    if hasattr(c, "random_non_queries"):
        c = c.model_copy(update={"random_non_queries": True})
    seg = c.build(seed=3)
    x, y = seg.inputs[0].tolist(), seg.labels[0].tolist()
    roles = []
    if game in ("cumulative_parity", "cumulative_majority"):
        roles = ["bit"] * len(x)
    else:
        kv = small["num_kv_pairs"]
        per = 3 if game == "compositional_mqar" else 2
        ctx = (kv + small.get("num_updates", 0)) * per
        for i in range(len(x)):
            if i < ctx:
                roles.append("valor" if (i % per) == per - 1 else "clave")
            elif y[i] != -100 or (game == "compositional_mqar" and i + 1 < len(y) and y[i + 1] != -100):
                roles.append("pregunta")
            else:
                roles.append("relleno")
    return x, y, roles, None


def render_tokens(x, y, roles, display=None) -> str:
    chips = []
    display = display or [str(t) for t in x]
    for t, lab, r, d in zip(x, y, roles, display):
        ans = f'<span class="ans">→{lab}</span>' if lab != -100 and r != "bit" else ""
        sub = f'<span class="sub">{lab}</span>' if r == "bit" else ""
        chips.append(f'<span class="tok {r}" title="rol: {ROLE_LABEL[r]}">{html.escape(d)}{ans}{sub}</span>')
    return '<div class="seq">' + "".join(chips) + "</div>"


def knobs_table(game_cfg) -> str:
    def row(c):
        d = c.model_dump()
        parts = [f"{d['input_seq_len']} tokens"]
        if "num_kv_pairs" in d:
            parts.append(f"{d['num_kv_pairs']} pares")
        if "num_updates" in d:
            parts.append(f"{d['num_updates']} actualizaciones")
        if "str_len" in d:
            parts.append(f"{d['num_demos']} ejemplos en contexto · tiras de {d['str_len']} · profundidad {d['depth']} · {d['split']}")
        elif "width_min" in d:
            src = "instancias reales de 1D-ARC" if d["source"] == "1darc" else f"filas de {d['width_min']} a {d['width_max']} píxeles · reglas: {', '.join(d['rules'])}"
            parts.append(f"{d['num_demos']} ejemplos en contexto · {src} · {d['split']}")
        parts.append(f"{d['num_examples']:,} ejemplos".replace(",", "."))
        return " · ".join(parts)
    tr = "".join(f"<li>{row(c)}</li>" for c in game_cfg["train"])
    te = "".join(f"<li>{row(c)}</li>" for c in game_cfg["test"])
    return f'<div class="knobs"><div><h4>Entrena con</h4><ul>{tr}</ul></div><div><h4>Examina con</h4><ul>{te}</ul></div></div>'


# ---------- arquitecturas ----------

def family_of(name: str) -> str:
    for f in ("sliding_window", "gated_delta_net", "attention"):
        if name.startswith(f):
            return f
    return name


def arch_svg(model, info: dict | None) -> str:
    """Pila de capas de un ModelConfig de Zoology (receta de 2 capas: conv corta + mixer)."""
    fam = family_of(model.name)
    label, color = FAMILY[fam]
    mixers = model.sequence_mixer.kwargs["configs"]
    kw = mixers[-1]["kwargs"]
    if fam == "attention":
        note = "compara con todos los tokens anteriores · el KV cache crece con el largo"
    elif fam == "sliding_window":
        note = f"mira solo los últimos {kw['block_size']} tokens"
    else:
        note = f"estado S de tamaño fijo · {kw['num_heads']} heads" + (" · β∈(0,2): autovalores negativos" if kw.get("allow_neg_eigval") else "")
    state = f"{info['state_size']:,} bytes".replace(",", ".") if info else "—"
    params = f"{info['num_parameters']:,}".replace(",", ".") if info else "—"
    W, H = 460, 250
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Arquitectura {model.name} d{model.d_model}: embedding, convolución corta, {label}, predicción">']
    s.append('<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>')
    s.append('<line x1="110" y1="244" x2="110" y2="12" stroke="currentColor" stroke-width="1.2" marker-end="url(#ar)"/>')
    boxes = [("tokens → embedding", [f"d = {model.d_model}"], None, 196),
             ("capa 0: convolución corta", ["mezcla cada token con sus 2 vecinos"], None, 140),
             (f"capa 1: {label}", [n.strip() for n in note.split("·")], color, 84),
             ("predicción del token siguiente", [f"vocabulario de {model.vocab_size:,}".replace(",", ".")], None, 28)]
    for title, lines, col, y in boxes:
        fill = f'style="fill:{col};fill-opacity:0.14" stroke="{col}"' if col else 'style="fill:var(--card)" stroke="currentColor" stroke-opacity="0.45"'
        s.append(f'<rect x="20" y="{y}" width="180" height="40" rx="5" {fill} stroke-width="1.2"/>')
        s.append(f'<text x="110" y="{y+24}" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor">{html.escape(title)}</text>')
        y0 = y + 20 - 6.5 * (len(lines) - 1)
        for i, ln in enumerate(lines):
            s.append(f'<text x="214" y="{y0 + 13*i + 4:.0f}" font-size="10.5" fill="currentColor" fill-opacity="0.75">{html.escape(ln)}</text>')
    s.append(f'<text x="20" y="246" font-size="10.5" fill="currentColor" fill-opacity="0.75">estado al generar: {state} · parámetros: {params}</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- resultados ----------

def marker_svg(kind: int, cx: float, cy: float, color: str, title: str) -> str:
    """Marcador por variante dentro de la familia: círculo, cuadrado, rombo (≥8px)."""
    t = f"<title>{title}</title>"
    common = f'fill="{color}" stroke="var(--card)" stroke-width="1.5"'
    if kind % 3 == 0:
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" {common}>{t}</circle>'
    if kind % 3 == 1:
        return f'<rect x="{cx-4:.1f}" y="{cy-4:.1f}" width="8" height="8" {common}>{t}</rect>'
    return f'<polygon points="{cx:.1f},{cy-5.5:.1f} {cx+5.5:.1f},{cy:.1f} {cx:.1f},{cy+5.5:.1f} {cx-5.5:.1f},{cy:.1f}" {common}>{t}</polygon>'


def svg_line_chart(series: list[tuple[str, str, str, int, list[tuple[float, float]]]], xs: list[float], title: str, xlabel: str) -> str:
    """series: (nombre, color, dasharray, marcador, puntos). Eje x en log2 con ticks en xs."""
    W, H, L, Rm, T, B = 640, 340, 56, 16, 36, 48
    pw, ph = W - L - Rm, H - T - B
    lx = [math.log2(x) for x in xs]
    x0, x1 = min(lx), max(lx)
    def X(x): return L + (math.log2(x) - x0) / (x1 - x0 or 1) * pw
    def Y(y): return T + (1 - y) * ph
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    out.append(f'<text x="{L}" y="20" font-size="13" font-weight="600" fill="currentColor">{html.escape(title)}</text>')
    for yt in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" y1="{Y(yt):.1f}" x2="{W-Rm}" y2="{Y(yt):.1f}" stroke="currentColor" stroke-opacity="0.12"/>')
        out.append(f'<text x="{L-8}" y="{Y(yt)+4:.1f}" text-anchor="end" font-size="10" fill="currentColor" fill-opacity="0.7">{yt:g}</text>')
    for x in xs:
        out.append(f'<text x="{X(x):.1f}" y="{H-B+16}" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">{int(x)}</text>')
    out.append(f'<text x="{L + pw/2:.1f}" y="{H-10}" text-anchor="middle" font-size="11" fill="currentColor" fill-opacity="0.75">{html.escape(xlabel)}</text>')
    out.append(f'<text transform="translate(14,{T + ph/2:.0f}) rotate(-90)" text-anchor="middle" font-size="11" fill="currentColor" fill-opacity="0.75">precisión</text>')
    for name, color, dash, mk, pts in series:
        pts = [(x, y) for x, y in pts if y == y]  # sin NaN
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
        for x, y in pts:
            out.append(marker_svg(mk, X(x), Y(y), color, f"{html.escape(name)} · x={int(x)} · precisión {y:.2f}"))
    out.append("</svg>")
    return "".join(out)


def results_section(df: pd.DataFrame | None, games: dict, model_info: dict, n_total: int = 0, n_done: int = 0) -> str:
    if df is None or df.empty:
        return '<p class="muted">Todavía no hay resultados en esta carpeta. Esta sección se llena sola cuando termina el banco.</p>'
    parts = []
    if n_total and n_done < n_total:
        parts.append(f'<p class="muted"><strong>Parcial:</strong> hay resultados de {n_done} de {n_total} corridas; el banco sigue corriendo. Las curvas pueden cambiar.</p>')
    seen_marker = {}
    dashes = {"attention": "", "sliding_window": "", "gated_delta_net": ""}
    for game in games:
        g = df[df.game == game]
        if g.empty:
            continue
        title, _, xkey = GAME_TEXT.get(game, (game, "", None))
        if xkey is None:
            cols = sorted([c for c in g.columns if c.startswith("valid/split/accuracy-") and g[c].notna().any()])
            xs = []
        else:
            cols = sorted([c for c in g.columns if c.startswith(f"valid/{xkey}/accuracy-") and g[c].notna().any()], key=lambda c: float(c.rsplit("-", 1)[1]))
            xs = [float(c.rsplit("-", 1)[1]) for c in cols]
        series, per_family = [], {}
        for _, row in g.sort_values("arch").iterrows():
            fam = family_of(row["arch"])
            mk = per_family.get(fam, 0); per_family[fam] = mk + 1
            series.append((row["arch"], FAMILY[fam][1], "6 4" if "_neg" in row["arch"] else "", mk, [(x, row[c]) for x, c in zip(xs, cols)]))
        if xs:
            chart = svg_line_chart(series, xs, title, "pares clave→valor por secuencia" if xkey == "num_kv_pairs" else "largo de la secuencia (tokens)")
        else:
            chart = f'<h3>{html.escape(title)}</h3>'
        legend = "".join(f'<li><svg width="26" height="12" viewBox="0 0 26 12" aria-hidden="true"><line x1="0" y1="6" x2="26" y2="6" stroke="{c}" stroke-width="2" stroke-dasharray="{d}"/>{marker_svg(mk, 13, 6, c, "")}</svg>{html.escape(n)}</li>' for n, c, d, mk, _ in series)
        tbl = g[["arch", "valid/accuracy"] + cols + ["state_bytes", "epochs", "minutes"]].copy()
        tbl.columns = ["arquitectura", "promedio"] + [c.rsplit("-", 1)[1] for c in cols] + ["estado (bytes)", "épocas", "min"]
        acc_cols = set(range(1, 2 + len(cols)))  # promedio + dificultades
        def fmt(i, v):
            if isinstance(v, str): return html.escape(v)
            if i in acc_cols: return f"{v:.2f}"
            if isinstance(v, float) and not v.is_integer(): return f"{v:.1f}"
            return f"{int(v):,}".replace(",", ".")
        rows = "".join("<tr>" + "".join(f"<td>{fmt(i, v)}</td>" for i, v in enumerate(r)) + "</tr>" for r in tbl.itertuples(index=False))
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in tbl.columns)
        parts.append(f'<div class="result"><div class="chart">{chart}</div>{"<ul class=legend>" + legend + "</ul>" if xs else ""}<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div></div>')
    return "".join(parts)


# ---------- página ----------

CSS = """
:root{--bg:#f3f5f7;--card:#ffffff;--fg:#1a2430;--muted:#5d6b78;--line:#d3dae1;--clave:#dbe7f8;--clave-b:#2a78d6;--valor:#dff3ec;--valor-b:#1baf7a;--pregunta:#fde6da;--pregunta-b:#eb6834;--relleno:#eef0f2;--relleno-b:#c3cad1;--bit:#ece9f7;--bit-b:#7a6ea8}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#10161c;--card:#171f27;--fg:#e7ecf1;--muted:#9aa8b5;--line:#3a4652;--clave:#1e3350;--valor:#173a32;--pregunta:#4a2a1c;--relleno:#232b33;--relleno-b:#4a5663;--bit:#2a2640}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#10161c;--card:#171f27;--fg:#e7ecf1;--muted:#9aa8b5;--line:#3a4652;--clave:#1e3350;--valor:#173a32;--pregunta:#4a2a1c;--relleno:#232b33;--relleno-b:#4a5663;--bit:#2a2640}
body{background:var(--bg);color:var(--fg);font-family:"Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif;font-size:16px;line-height:1.5;padding-block:28px 60px;padding-inline:16px}
main{max-width:1040px;margin:0 auto;display:grid;gap:36px}
h1{font-size:28px;margin:0 0 6px;text-wrap:balance}h2{font-size:21px;margin:0 0 10px}h3{font-size:17px;margin:0 0 6px}h4{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 4px}
p{margin:0;max-width:70ch}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:16px 18px}
.games{display:grid;gap:14px}.game{display:grid;gap:10px}
.seq{display:flex;flex-wrap:wrap;gap:4px;font-family:"JetBrains Mono",ui-monospace,Menlo,monospace;font-size:12px}
.tok{display:inline-flex;flex-direction:column;align-items:center;min-width:30px;padding:3px 5px;border-radius:4px;border:1px solid var(--relleno-b);background:var(--relleno);line-height:1.2}
.tok.clave{background:var(--clave);border-color:var(--clave-b)}.tok.valor{background:var(--valor);border-color:var(--valor-b)}.tok.pregunta{background:var(--pregunta);border-color:var(--pregunta-b);font-weight:700}.tok.bit{background:var(--bit);border-color:var(--bit-b)}
.ans{font-size:10px;color:var(--pregunta-b);font-weight:700}.sub{font-size:10px;color:var(--bit-b)}
.rolelegend{display:flex;flex-wrap:wrap;gap:10px 16px;font-size:13px;color:var(--muted)}.rolelegend span{display:inline-flex;align-items:center;gap:6px}.rolelegend i{width:14px;height:14px;border-radius:3px;border:1px solid;display:inline-block}
.knobs{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}.knobs ul{margin:0;padding-left:18px;font-size:14px;color:var(--muted)}
.archs{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}.arch svg{width:100%;height:auto;color:var(--fg)}
.proto{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}.proto ul{margin:0;padding-left:18px}
.result{display:grid;gap:10px;margin-bottom:22px}.chart svg{width:100%;max-width:640px;height:auto;color:var(--fg)}
.legend{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--muted)}.legend .sw{display:inline-block;width:14px;height:4px;border-radius:2px;margin-right:6px;vertical-align:middle}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:4px 8px;text-align:right;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
"""


def build(exp_path: Path, bank: Path) -> Path:
    spec = importlib.util.spec_from_file_location("exp", exp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    games, L = mod.GAMES, 1024
    try:
        df = scorecard(bank)
    except SystemExit:
        df = None
    # info de modelos (estado, parámetros) desde los resultados, si los hay
    model_info = {}
    for mj in bank.glob("*/*/model.json"):
        cfg = json.loads((mj.parent / "config.json").read_text())
        model_info[f'{cfg["model"]["name"]}-d{cfg["model"]["d_model"]}'] = json.loads(mj.read_text())
    archs = mod.architectures(L, mod.VOCAB_SIZE)

    games_html = []
    for game, gcfg in games.items():
        title, desc, _ = GAME_TEXT[game]
        x, y, roles, display = example_roles(game, gcfg["test"][0])
        games_html.append(f'<div class="card game"><h3>{title}</h3><p>{desc}</p>{render_tokens(x, y, roles, display)}{knobs_table(gcfg)}</div>')
    GAME_TEXT.setdefault("rules", ("Reglas en contexto (mini-ARC)", "", None))
    rolelegend = '<div class="rolelegend">' + "".join(f'<span><i style="background:var(--{r});border-color:var(--{r}-b)"></i>{ROLE_LABEL[r]}</span>' for r in ("clave", "valor", "pregunta", "relleno", "bit")) + '<span>→ valor esperado en la pregunta · número chico debajo del bit: respuesta esperada</span></div>'
    archs_html = "".join(f'<div class="card arch"><h3>{html.escape(m.name)} · d{m.d_model}</h3>{arch_svg(m, model_info.get(f"{m.name}-d{m.d_model}"))}</div>' for m in archs)
    n_runs = len(mod.configs)
    proto = f'''<div class="proto">
      <div class="card"><h4>Fijo en todas las corridas</h4><ul><li>Los mismos datos de entrenamiento y examen por juego (misma semilla)</li><li>2 capas: convolución corta + la capa que se prueba, sin MLP</li><li>Hasta {mod.MAX_EPOCHS} épocas, corte anticipado al llegar a 99% de precisión</li><li>Batch 256, AdamW, weight decay 0.1, cosine</li></ul></div>
      <div class="card"><h4>Lo que se barre</h4><ul><li>{len(archs)} arquitecturas (familia × tamaño)</li><li>{len(mod.LRS)} learning rates: {", ".join(f"{lr:.0e}" for lr in mod.LRS)}; se reporta el mejor</li><li>{len(games)} juegos, cada uno con varias dificultades</li><li>Total: {n_runs} corridas</li></ul></div>
      <div class="card"><h4>Qué se mide</h4><ul><li>Precisión en las preguntas, por dificultad (máximo sobre épocas)</li><li>Generalización: el examen incluye largos y cantidades de pares mayores que el entrenamiento</li><li>Memoria del estado al generar (bytes) y parámetros</li><li>Épocas y minutos de entrenamiento</li></ul></div>
    </div>'''
    n_done = len(list(bank.glob("*/*/metrics.jsonl")))
    results = results_section(df, games, model_info, n_total=n_runs, n_done=n_done)

    page = f'''<title>Tablero del banco</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap">
<style>{CSS}</style>
<main>
  <header><h1>Tablero del banco v1</h1><p class="muted">Entra una arquitectura, se entrena desde cero en cada juego y sale una tarjeta de puntajes. Esta página se regenera desde la config del experimento y la carpeta de resultados.</p></header>
  <section><h2>1. Los juegos</h2><p>Cada juego es un script que inventa secuencias. Abajo, una secuencia real de ejemplo de cada uno (achicada para que entre), con cada token pintado según su rol.</p>{rolelegend}<div class="games">{"".join(games_html)}</div></section>
  <section><h2>2. Las arquitecturas</h2><p>Todas comparten la misma receta de dos capas; lo único que cambia es la capa 1, resaltada en color. El estado es lo que el modelo carga mientras genera: crece con el largo en atención, es fijo en Gated DeltaNet y acotado por la ventana en la ventana deslizante.</p><div class="archs">{archs_html}</div></section>
  <section><h2>3. El protocolo</h2>{proto}</section>
  <section><h2>4. Los resultados</h2><p>Una curva por arquitectura: precisión contra dificultad. Pasá el mouse por los puntos para ver el valor exacto. Debajo, la tabla con el mejor learning rate de cada arquitectura.</p>{results}</section>
</main>'''
    out = bank / "tablero.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    exp = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/bench_v1.py")
    bank = Path(sys.argv[2] if len(sys.argv) > 2 else "results/bench_v1")
    print("tablero en", build(exp, bank))
