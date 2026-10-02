"""Banco de pruebas v1: cada arquitectura se entrena desde cero en cada juego.

Juegos (todos ya implementados en Zoology; solo elegimos las perillas):
  mqar                  recuerdo exacto: pares clave→valor, varias preguntas
  forgetting_mqar       actualización: una clave cambia de valor, vale el último
  compositional_mqar    claves compuestas: el valor depende de DOS tokens de clave
  cumulative_parity     seguimiento de estado: paridad acumulada de una tira de bits
  cumulative_majority   conteo: mayoría acumulada de una tira de bits
  rules                 mini-ARC: inducir una regla nueva de K ejemplos en contexto y aplicarla;
                        examen con composiciones y primitivas nunca vistas
  arc1d                 mini-ARC con fondo y objetos (catálogo de 1D-ARC); examen con reglas
                        nunca vistas y con las instancias reales del dataset 1D-ARC
En todos, el examen incluye secuencias más largas que las de entrenamiento
(generalización de largo). La tarjeta de puntajes la arma rnn_lookup.scorecard.

Arquitecturas: atención (d 64, 128), ventana (d 128; w 16, 64),
Gated DeltaNet (d 64, 128, 256) y su variante con autovalores negativos (d 128).
3 learning rates. 8 x 3 x 7 = 168 corridas.

Variables de entorno:
  BENCH_GAMES=mqar,cumulative_parity   correr solo algunos juegos
  BENCH_SMOKE=1                        versión mínima (pocos ejemplos, 2 épocas) para probar la plomería
  ZOOLOGY_CACHE=...                    dónde cachear los datos generados

    ZOOLOGY_WORKERS_PER_GPU=4 python -m zoology.launch experiments/bench_v1.py -p
"""
import os
import numpy as np
from zoology.config import TrainConfig, DataConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.data.forgetting_mqar import ForgettingMQARConfig
from zoology.data.compositional_mqar import CompositionalMQARConfig
from zoology.data.circuits import CumulativeParityConfig, CumulativeMajorityConfig
from rnn_lookup.games.rules import RulesConfig
from rnn_lookup.games.arc1d import Arc1DConfig
from rnn_lookup.zoo import attention, sliding_window, gated_delta_net, gated_delta_net_neg, VOCAB_SIZE

SMOKE = os.environ.get("BENCH_SMOKE") == "1"
N_TRAIN_BIG = 2_000 if SMOKE else 100_000
N_TRAIN = 2_000 if SMOKE else 20_000
N_TEST = 200 if SMOKE else 1_000
MAX_EPOCHS = 2 if SMOKE else 32
LRS = [1e-3] if SMOKE else [float(x) for x in np.logspace(-3, -2, 3)]  # 1e-3, 3.2e-3, 1e-2
BIT_VOCAB = 16  # paridad y mayoría usan solo los tokens 0 y 1
RULES_VOCAB, RULES_L = 16, 96  # 8 símbolos + 3 especiales; largo para K=5 ejemplos de tiras de 6
RULES_PRIMS = ["reverse", "rot1", "rot2", "sort", "incr", "decr", "first_to_all"]  # mirror_half queda afuera
RULES_HELD_OUT = [("reverse", "rot1"), ("sort", "incr"), ("rot2", "decr")]
ARC_L = 288  # K=3 ejemplos con filas de hasta 33 píxeles (cubre 860 de las 901 instancias reales)
ARC_SEEN = ["move", "fill", "hollow", "flip", "recolor_size", "denoise", "recolor_parity"]
ARC_UNSEEN = ["pattern_copy", "scale"]
OUTPUT_DIR = "results/bench_v1_smoke" if SMOKE else "results/bench_v1"


def mqar(L, kv, n):
    return MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=L, num_kv_pairs=kv, num_examples=n)


def fmqar(L, kv, upd, n):
    return ForgettingMQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=L, num_kv_pairs=kv, num_updates=upd, num_examples=n)


def cmqar(L, kv, n):
    return CompositionalMQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=L, num_kv_pairs=kv, num_examples=n)


def arc(n, split, **kw):
    return Arc1DConfig(vocab_size=16, input_seq_len=ARC_L, num_examples=n, split=split, **kw)


def rule(n, depth, split, demos=3, **kw):
    return RulesConfig(vocab_size=RULES_VOCAB, input_seq_len=RULES_L, num_symbols=8, str_len=6, num_demos=demos,
                       num_examples=n, depth=depth, split=split, **kw)


GAMES = {
    "mqar": dict(
        vocab=VOCAB_SIZE,
        slices=["num_kv_pairs", "input_seq_len"],
        train=[mqar(64, 4, N_TRAIN_BIG), mqar(128, 8, N_TRAIN), mqar(256, 16, N_TRAIN), mqar(256, 32, N_TRAIN), mqar(256, 64, N_TRAIN)],
        test=[mqar(64, 4, N_TEST), mqar(64, 8, N_TEST), mqar(64, 16, N_TEST), mqar(128, 32, N_TEST),
              mqar(256, 64, N_TEST), mqar(512, 128, N_TEST), mqar(1024, 256, N_TEST)],
    ),
    "forgetting_mqar": dict(
        vocab=VOCAB_SIZE,
        slices=["num_kv_pairs", "input_seq_len"],
        train=[fmqar(128, 8, 4, N_TRAIN), fmqar(256, 16, 8, N_TRAIN), fmqar(256, 32, 16, N_TRAIN)],
        test=[fmqar(128, 8, 4, N_TEST), fmqar(256, 16, 8, N_TEST), fmqar(256, 32, 16, N_TEST),
              fmqar(512, 64, 32, N_TEST), fmqar(1024, 128, 64, N_TEST)],
    ),
    "compositional_mqar": dict(
        vocab=VOCAB_SIZE,
        slices=["num_kv_pairs", "input_seq_len"],
        train=[cmqar(128, 4, N_TRAIN), cmqar(128, 9, N_TRAIN), cmqar(128, 16, N_TRAIN), cmqar(256, 36, N_TRAIN)],
        test=[cmqar(128, 4, N_TEST), cmqar(128, 9, N_TEST), cmqar(128, 16, N_TEST), cmqar(256, 36, N_TEST),
              cmqar(512, 64, N_TEST), cmqar(1024, 144, N_TEST)],
    ),
    "cumulative_parity": dict(
        vocab=BIT_VOCAB,
        slices=["input_seq_len"],
        train=[CumulativeParityConfig(vocab_size=BIT_VOCAB, input_seq_len=64, num_examples=N_TRAIN)],
        test=[CumulativeParityConfig(vocab_size=BIT_VOCAB, input_seq_len=L, num_examples=N_TEST) for L in (64, 128, 256, 512)],
    ),
    "cumulative_majority": dict(
        vocab=BIT_VOCAB,
        slices=["input_seq_len"],
        train=[CumulativeMajorityConfig(vocab_size=BIT_VOCAB, input_seq_len=64, num_examples=N_TRAIN)],
        test=[CumulativeMajorityConfig(vocab_size=BIT_VOCAB, input_seq_len=L, num_examples=N_TEST) for L in (64, 128, 256, 512)],
    ),
    "rules": dict(
        vocab=RULES_VOCAB,
        slices=["split", "depth"],
        train=[rule(N_TRAIN_BIG if not SMOKE else N_TRAIN, 1, "seen", primitives=RULES_PRIMS),
               rule(N_TRAIN_BIG if not SMOKE else N_TRAIN, 2, "seen", primitives=RULES_PRIMS, exclude=RULES_HELD_OUT)],
        test=[rule(N_TEST, 1, "seen", primitives=RULES_PRIMS),
              rule(N_TEST, 2, "seen", primitives=RULES_PRIMS, exclude=RULES_HELD_OUT),
              rule(N_TEST, 2, "unseen_comp", compositions=RULES_HELD_OUT),
              rule(N_TEST, 1, "unseen_prim", compositions=[("mirror_half",)]),
              rule(N_TEST, 1, "seen_5demos", demos=5, primitives=RULES_PRIMS)],
    ),
    "arc1d": dict(
        vocab=16,
        slices=["split"],
        train=[arc(N_TRAIN_BIG if not SMOKE else N_TRAIN, "seen", rules=ARC_SEEN)],
        test=[arc(N_TEST, "seen", rules=ARC_SEEN),
              arc(N_TEST, "unseen_rule", rules=ARC_UNSEEN),
              arc(10_000, "1darc_real", source="1darc")],
    ),
}

selected = os.environ.get("BENCH_GAMES")
if selected:
    GAMES = {g: GAMES[g] for g in selected.split(",")}


def architectures(L, vocab):
    return (
        [attention(d, L, vocab) for d in (64, 128)]
        + [sliding_window(128, w, L, vocab) for w in (16, 64)]
        + [gated_delta_net(d, L, vocab) for d in (64, 128, 256)]
        + [gated_delta_net_neg(128, L, vocab)]  # variante con autovalores negativos
    )


configs = []
for game, g in GAMES.items():
    L = max(c.input_seq_len for c in g["train"] + g["test"])
    data = DataConfig(
        train_configs=g["train"],
        test_configs=g["test"],
        batch_size=(256, 32),
        cache_dir=os.environ.get("ZOOLOGY_CACHE", "data/zoology-cache"),
    )
    for model in architectures(L, g["vocab"]):
        for lr in LRS:
            configs.append(
                TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=lr,
                    max_epochs=MAX_EPOCHS,
                    slice_keys=g["slices"],
                    sweep_id=game,
                    run_id=f"{model.name}-d{model.d_model}-lr{lr:.1e}",
                    output_dir=OUTPUT_DIR,
                )
            )
