"""Juego `arc1d`: mini-ARC con fondo y objetos, siguiendo el catálogo de 1D-ARC.

Cada "imagen" es una fila de píxeles con colores 1..9 sobre fondo 0. Un objeto es
un tramo contiguo de un mismo color. Cada secuencia trae una regla (una de las de
abajo, con sus parámetros sorteados por secuencia), K ejemplos entrada→salida y
una consulta; el modelo produce la salida de la consulta token a token.

    x1 → y1 ; x2 → y2 ; x3 → y3 ; xq → yq  PAD ...     (tokens: colores 0..9, →=10, ;=11, PAD=12)

Reglas del generador (todas tomadas de 1D-ARC, Xu et al. 2023):
    move            mover el objeto k píxeles a la derecha (k fijo por secuencia)
    fill            rellenar entre dos píxeles del mismo color
    hollow          vaciar el interior de un objeto, dejar los extremos
    flip            dar vuelta la fila
    recolor_size    recolorear cada objeto según su tamaño (mapa tamaño→color por secuencia)
    denoise         borrar los píxeles sueltos, dejar el objeto grande
    pattern_copy    copiar el patrón inicial (3 píxeles) centrado en cada marcador
    scale           estirar el objeto hasta el marcador de otro color
    recolor_parity  recolorear objetos alternando dos colores (impar/par)

`source="1darc"` carga en cambio las instancias reales del dataset 1D-ARC
(data/1D-ARC/dataset/<tipo>/*.json) con el mismo formato de tokens: sirve como
examen con datos que no escribimos nosotros.
"""
import glob
import json
from typing import List, Optional

import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment

ARROW, SEP, PAD = 10, 11, 12
RULES = ["move", "fill", "hollow", "flip", "recolor_size", "denoise", "pattern_copy", "scale", "recolor_parity"]


def _objects(row):
    """Tramos contiguos de color != 0: lista de (inicio, fin_exclusivo, color)."""
    out, i = [], 0
    while i < len(row):
        if row[i] != 0:
            j = i
            while j < len(row) and row[j] == row[i]:
                j += 1
            out.append((i, j, int(row[i])))
            i = j
        else:
            i += 1
    return out


def _place_objects(rng, W, sizes, color_fn, min_gap=1, start=0):
    """Coloca objetos de los tamaños dados, en orden, con huecos aleatorios >= min_gap."""
    total = sum(sizes) + min_gap * (len(sizes) - 1)
    if W - start < total:
        return None
    slack = W - start - total
    gaps = rng.multinomial(slack, np.ones(len(sizes) + 1) / (len(sizes) + 1))
    row = np.zeros(W, dtype=np.int64)
    pos = start + gaps[0]
    for idx, (sz, g) in enumerate(zip(sizes, gaps[1:])):
        row[pos:pos + sz] = color_fn(idx, sz)
        pos += sz + min_gap + g
    return row


# --- cada regla: sample_params(rng) y make_pair(rng, W, params) -> (entrada, salida) ---

def _move(rng, W, p):
    k, c = p["k"], p["c"]
    ell = int(rng.integers(2, 6))
    start = int(rng.integers(0, W - ell - k + 1))
    x = np.zeros(W, dtype=np.int64); x[start:start + ell] = c
    y = np.zeros(W, dtype=np.int64); y[start + k:start + k + ell] = c
    return x, y

def _fill(rng, W, p):
    c = p["c"]
    i = int(rng.integers(0, W - 4)); j = int(rng.integers(i + 3, W))
    x = np.zeros(W, dtype=np.int64); x[i] = c; x[j] = c
    y = x.copy(); y[i:j + 1] = c
    return x, y

def _hollow(rng, W, p):
    c = p["c"]; ell = int(rng.integers(3, min(9, W) + 1)); start = int(rng.integers(0, W - ell + 1))
    x = np.zeros(W, dtype=np.int64); x[start:start + ell] = c
    y = np.zeros(W, dtype=np.int64); y[start] = c; y[start + ell - 1] = c
    return x, y

def _flip(rng, W, p):
    n = int(rng.integers(1, 4))
    x = _place_objects(rng, W, [int(s) for s in rng.integers(1, 4, size=n)], lambda i, s: int(rng.integers(1, 10)))
    return x, x[::-1].copy()

def _recolor_size(rng, W, p):
    c, cmap = p["c"], p["size_map"]
    n = int(rng.integers(2, 5))
    sizes = [int(s) for s in rng.integers(1, 4, size=n)]
    x = _place_objects(rng, W, sizes, lambda i, s: c)
    if x is None:
        return None
    y = x.copy()
    for a, b, _ in _objects(x):
        y[a:b] = cmap[b - a]
    return x, y

def _denoise(rng, W, p):
    c = p["c"]; ell = int(rng.integers(3, 7))
    x = np.zeros(W, dtype=np.int64)
    start = int(rng.integers(0, W - ell + 1)); x[start:start + ell] = c
    free = [i for i in range(W) if x[max(0, i - 1):i + 2].sum() == 0]
    noise = rng.choice(free, size=min(len(free), int(rng.integers(1, 4))), replace=False)
    for i in noise:
        if x[max(0, i - 1):i + 2].sum() == 0:
            x[i] = c
    y = np.zeros(W, dtype=np.int64); y[start:start + ell] = c
    return x, y

def _pattern_copy(rng, W, p):
    c = p["c"]
    x = np.zeros(W, dtype=np.int64); x[0:3] = c
    n = int(rng.integers(1, 4))
    pos, marks = 5, []
    for _ in range(n):
        pos += int(rng.integers(0, 3))
        if pos + 1 >= W:
            break
        marks.append(pos); pos += 4
    if not marks:
        return None
    for m in marks:
        x[m] = c
    y = x.copy()
    for m in marks:
        y[m - 1:m + 2] = c
    return x, y

def _scale(rng, W, p):
    c, d = p["c"], p["d"]
    ell = int(rng.integers(2, 5)); start = int(rng.integers(0, W - ell - 3))
    marker = int(rng.integers(start + ell + 2, W))
    x = np.zeros(W, dtype=np.int64); x[start:start + ell] = c; x[marker] = d
    y = x.copy(); y[start:marker] = c
    return x, y

def _recolor_parity(rng, W, p):
    c, a, b = p["c"], p["a"], p["b"]
    n = int(rng.integers(2, 6))
    x = _place_objects(rng, W, [int(s) for s in rng.integers(1, 4, size=n)], lambda i, s: c)
    if x is None:
        return None
    y = x.copy()
    for idx, (s0, s1, _) in enumerate(_objects(x)):
        y[s0:s1] = a if idx % 2 == 0 else b
    return x, y


def _params(rng, rule):
    c = int(rng.integers(1, 10))
    others = [k for k in range(1, 10) if k != c]
    if rule == "move":
        return {"k": int(rng.integers(1, 4)), "c": c}
    if rule == "recolor_size":
        cols = rng.choice(others, size=3, replace=False)
        return {"c": c, "size_map": {1: int(cols[0]), 2: int(cols[1]), 3: int(cols[2])}}
    if rule == "scale":
        return {"c": c, "d": int(rng.choice(others))}
    if rule == "recolor_parity":
        a, b = rng.choice(others, size=2, replace=False)
        return {"c": c, "a": int(a), "b": int(b)}
    return {"c": c}


MAKERS = {"move": _move, "fill": _fill, "hollow": _hollow, "flip": _flip, "recolor_size": _recolor_size,
          "denoise": _denoise, "pattern_copy": _pattern_copy, "scale": _scale, "recolor_parity": _recolor_parity}


def _tokens(pairs, query, L):
    toks = []
    for x, y in pairs:
        toks += list(x) + [ARROW] + list(y) + [SEP]
    xq, yq = query
    toks += list(xq) + [ARROW] + list(yq)
    if len(toks) > L:
        return None
    inp = np.full(L, PAD, dtype=np.int64); inp[:len(toks)] = toks
    lab = np.full(L, -100, dtype=np.int64)
    n = len(yq); end = len(toks)
    lab[end - n - 1:end - 1] = yq
    return inp, lab


class Arc1DConfig(DataSegmentConfig):
    name: str = "arc1d"
    source: str = "gen"                 # "gen" (nuestro generador) o "1darc" (dataset real)
    rules: List[str] = RULES            # reglas permitidas (source=gen)
    num_demos: int = 3
    width_min: int = 12
    width_max: int = 24
    task_types: Optional[List[str]] = None   # source=1darc: carpetas a cargar (None = todas)
    dataset_dir: str = "data/1D-ARC/dataset"
    split: str = "seen"

    def build(self, seed: int) -> DataSegment:
        return arc1d(**self.model_dump(), seed=seed)


def arc1d(vocab_size, num_examples, input_seq_len, seed, source="gen", rules=RULES, num_demos=3,
          width_min=12, width_max=24, task_types=None, dataset_dir="data/1D-ARC/dataset", split="seen", **kw) -> DataSegment:
    assert vocab_size >= 13
    L = input_seq_len
    inputs, labels = [], []
    if source == "1darc":
        files = []
        for t in (task_types or sorted(d.split("/")[-1] for d in glob.glob(f"{dataset_dir}/*"))):
            files += sorted(glob.glob(f"{dataset_dir}/{t}/*.json"))
        skipped = 0
        for f in files:
            d = json.load(open(f))
            pairs = [(np.array(ex["input"][0]), np.array(ex["output"][0])) for ex in d["train"]]
            q = (np.array(d["test"][0]["input"][0]), np.array(d["test"][0]["output"][0]))
            tk = _tokens(pairs, q, L)
            if tk is None:
                skipped += 1; continue
            inputs.append(tk[0]); labels.append(tk[1])
        if skipped:
            print(f"arc1d/1darc: {skipped} instancias no entran en {L} tokens y se omiten")
        inputs, labels = inputs[:num_examples], labels[:num_examples]
    else:
        rng = np.random.default_rng(seed)
        for rule in rules:
            assert rule in MAKERS, f"regla desconocida: {rule}"
        while len(inputs) < num_examples:
            rule = rules[rng.integers(len(rules))]
            p = _params(rng, rule)
            W = int(rng.integers(width_min, width_max + 1))
            pairs = []
            ok = True
            for _ in range(num_demos + 1):
                pair = None
                for _try in range(20):
                    pair = MAKERS[rule](rng, W, p)
                    if pair is not None:
                        break
                if pair is None:
                    ok = False; break
                pairs.append(pair)
            if not ok:
                continue
            tk = _tokens(pairs[:-1], pairs[-1], L)
            if tk is None:
                continue
            inputs.append(tk[0]); labels.append(tk[1])
    return DataSegment(
        torch.tensor(np.stack(inputs)), torch.tensor(np.stack(labels)),
        slices={"split": split, "source": source, "num_demos": num_demos, "input_seq_len": L},
    )
