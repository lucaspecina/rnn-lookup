"""Banco de pruebas v1: cada arquitectura se entrena desde cero en cada juego.

Juegos (todos ya implementados en Zoology; solo elegimos las perillas):
  mqar                  recuerdo exacto: pares clave→valor, varias preguntas
  forgetting_mqar       actualización: una clave cambia de valor, vale el último
  compositional_mqar    claves compuestas: el valor depende de DOS tokens de clave
  cumulative_parity     seguimiento de estado: paridad acumulada de una tira de bits
  cumulative_majority   conteo: mayoría acumulada de una tira de bits
En todos, el examen incluye secuencias más largas que las de entrenamiento
(generalización de largo). La tarjeta de puntajes la arma rnn_lookup.scorecard.

Arquitecturas: atención (d 64, 128), ventana (d 128; w 16, 64),
Gated DeltaNet (d 64, 128, 256) y su variante con autovalores negativos (d 128).
3 learning rates. 8 x 3 x 5 = 120 corridas.

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
from rnn_lookup.zoo import attention, sliding_window, gated_delta_net, gated_delta_net_neg, VOCAB_SIZE

SMOKE = os.environ.get("BENCH_SMOKE") == "1"
N_TRAIN_BIG = 2_000 if SMOKE else 100_000
N_TRAIN = 2_000 if SMOKE else 20_000
N_TEST = 200 if SMOKE else 1_000
MAX_EPOCHS = 2 if SMOKE else 32
LRS = [1e-3] if SMOKE else [float(x) for x in np.logspace(-3, -2, 3)]  # 1e-3, 3.2e-3, 1e-2
BIT_VOCAB = 16  # paridad y mayoría usan solo los tokens 0 y 1


def mqar(L, kv, n):
    return MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=L, num_kv_pairs=kv, num_examples=n)


def fmqar(L, kv, upd, n):
    return ForgettingMQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=L, num_kv_pairs=kv, num_updates=upd, num_examples=n)


def cmqar(L, kv, n):
    return CompositionalMQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=L, num_kv_pairs=kv, num_examples=n)


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
                    output_dir="results/bench_v1",
                )
            )
