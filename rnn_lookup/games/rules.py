"""Juego "reglas en contexto" (mini-ARC en secuencias).

Cada secuencia trae una regla NUEVA, K ejemplos de entrada→salida y una consulta.
El modelo tiene que producir la salida de la consulta token a token. La regla es
una composición de 1 a 3 primitivas sobre tiras de símbolos, y cambia en cada
secuencia: no sirve memorizarla, hay que inducirla de los ejemplos.

Layout de una secuencia (n = largo de las tiras, K = ejemplos):

    x1 → y1 ; x2 → y2 ; ... ; xK → yK ; xq → yq  PAD PAD ...
    etiquetas: -100 en todos lados salvo en los tokens de yq (predicción del
    siguiente token, como en MQAR), así que el puntaje es la precisión por token
    de la respuesta.

Perillas: K (num_demos), n (str_len), profundidad (depth), símbolos (num_symbols).
Para medir abstracción, cada config declara qué composiciones puede usar:
  - `primitives`: las primitivas permitidas (lista de nombres)
  - `compositions`: "all" o una lista explícita de tuplas; con `exclude` se
    sacan composiciones (para entrenar con unas y examinar con las que faltan)
El slice `split` queda en la metadata para separar en la tarjeta.
"""
from itertools import permutations
from typing import List, Optional, Tuple

import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment

# --- primitivas que conservan el largo ---------------------------------------
# cada una recibe (tira: np.ndarray de símbolos 0..S-1, S, params) y devuelve otra tira

def _reverse(s, S, p): return s[::-1].copy()
def _rot1(s, S, p): return np.roll(s, -1)
def _rot2(s, S, p): return np.roll(s, -2)
def _swap_pairs(s, S, p):  # intercambia de a pares vecinos; si n es impar, el último queda
    out = s.copy(); m = len(s) - len(s) % 2
    out[0:m:2], out[1:m:2] = s[1:m:2], s[0:m:2]
    return out
def _sort(s, S, p): return np.sort(s)
def _incr(s, S, p): return (s + 1) % S
def _decr(s, S, p): return (s - 1) % S
def _subst(s, S, p):  # a→b, (a, b) sorteados por secuencia: hay que inferirlos de los ejemplos
    a, b = p; out = s.copy(); out[s == a] = b; return out
def _mirror_half(s, S, p):  # la segunda mitad es la primera al revés
    n = len(s); h = n // 2; out = s.copy(); out[n - h:] = s[:h][::-1]; return out
def _first_to_all(s, S, p): return np.full_like(s, s[0])

PRIMITIVES = {
    "reverse": _reverse, "rot1": _rot1, "rot2": _rot2, "swap_pairs": _swap_pairs, "sort": _sort,
    "incr": _incr, "decr": _decr, "subst": _subst, "mirror_half": _mirror_half, "first_to_all": _first_to_all,
}


def all_compositions(primitives: List[str], depth: int) -> List[Tuple[str, ...]]:
    return [tuple(c) for c in permutations(primitives, depth)]


class RulesConfig(DataSegmentConfig):
    name: str = "rules"
    num_symbols: int = 8          # símbolos de las tiras (tokens 0..S-1)
    str_len: int = 6              # n
    num_demos: int = 3            # K
    depth: int = 1                # primitivas encadenadas por regla
    primitives: List[str] = ["reverse", "rot1", "rot2", "sort", "incr", "decr", "mirror_half", "first_to_all"]
    compositions: Optional[List[Tuple[str, ...]]] = None   # None = todas las permutaciones de `depth`
    exclude: List[Tuple[str, ...]] = []                     # composiciones que NO se usan (para held-out)
    split: str = "seen"           # etiqueta para la tarjeta: seen / unseen_comp / unseen_prim

    def build(self, seed: int) -> DataSegment:
        return rules(**self.model_dump(), seed=seed)


def rules(vocab_size, num_examples, input_seq_len, seed, num_symbols=8, str_len=6, num_demos=3, depth=1,
          primitives=None, compositions=None, exclude=(), split="seen", **kwargs) -> DataSegment:
    S, n, K = num_symbols, str_len, num_demos
    ARROW, SEP, PAD = S, S + 1, S + 2          # tokens especiales justo después de los símbolos
    assert vocab_size >= S + 3, "vocab_size tiene que cubrir símbolos + 3 especiales"
    need = K * (2 * n + 2) + 2 * n + 1
    assert need <= input_seq_len, f"input_seq_len={input_seq_len} < {need} necesario para K={K}, n={n}"
    comps = [tuple(c) for c in (compositions or all_compositions(primitives, depth))]
    excl = {tuple(c) for c in exclude}
    comps = [c for c in comps if c not in excl]
    assert comps, "no quedaron composiciones"
    for c in comps:
        for p in c:
            assert p in PRIMITIVES, f"primitiva desconocida: {p}"

    rng = np.random.default_rng(seed)
    inputs = np.full((num_examples, input_seq_len), PAD, dtype=np.int64)
    labels = np.full((num_examples, input_seq_len), -100, dtype=np.int64)
    for i in range(num_examples):
        comp = comps[rng.integers(len(comps))]
        params = {}
        if "subst" in comp:
            a, b = rng.choice(S, size=2, replace=False); params["subst"] = (int(a), int(b))
        def apply(s):
            for p in comp:
                s = PRIMITIVES[p](s, S, params.get(p))
            return s
        xs = rng.integers(0, S, size=(K + 1, n))
        toks = []
        for k in range(K):
            toks += list(xs[k]) + [ARROW] + list(apply(xs[k])) + [SEP]
        yq = list(apply(xs[K]))
        toks += list(xs[K]) + [ARROW] + yq
        L = len(toks)
        inputs[i, :L] = toks
        # etiqueta en la posición anterior a cada token de la respuesta (predicción del siguiente)
        start = L - n
        labels[i, start - 1:L - 1] = yq
    return DataSegment(
        torch.tensor(inputs), torch.tensor(labels),
        slices={"split": split, "depth": depth, "num_demos": K, "str_len": n, "input_seq_len": input_seq_len},
    )
