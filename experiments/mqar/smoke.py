"""Prueba de humo: una corrida chica de atención en MQAR, para validar la instalación.

En GPU tarda un minuto y debería llegar a accuracy alta. En CPU (Mac) solo sirve para
validar que todo corre: tarda ~5 min por época, mejor bajar num_examples.
    python -m zoology.launch experiments/mqar/smoke.py
Resultados en results/mqar/smoke/attention-smoke/{config.json,model.json,metrics.jsonl}
"""
from zoology.config import TrainConfig, DataConfig
from zoology.data.multiquery_ar import MQARConfig
from rnn_lookup.zoo import attention, VOCAB_SIZE

L = 64
data = DataConfig(
    train_configs=[MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=L, num_examples=20_000, num_kv_pairs=4)],
    test_configs=[MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=L, num_examples=500, num_kv_pairs=4)],
    batch_size=(64, 64),
)

configs = [
    TrainConfig(
        model=attention(d_model=64, input_seq_len=L),
        data=data,
        learning_rate=1e-3,
        max_epochs=10,
        slice_keys=["num_kv_pairs"],
        sweep_id="smoke",
        run_id="attention-smoke",
        output_dir="results/mqar",
    )
]
