"""Helpers para armar configs de Zoology sin repetir boilerplate.

Seguimos la "receta Based" que usa Zoology para comparar justo: cada modelo
tiene 2 capas, la primera es una convolución corta (BaseConv) y la segunda es
el mixer que queremos probar. Sin MLP (state_mixer = Identity), igual que en
zoology/experiments/mqar_example_configs/original_mqar_configs.py.
"""
from zoology.config import ModelConfig, ModuleConfig

VOCAB_SIZE = 8_192


def conv_mixer(input_seq_len: int) -> dict:
    return dict(
        name="zoology.mixers.base_conv.BaseConv",
        kwargs={"l_max": input_seq_len, "kernel_size": 3, "implicit_long_conv": True},
    )


def two_layer_model(name: str, d_model: int, mixer: dict, input_seq_len: int, vocab_size: int = VOCAB_SIZE, **extra) -> ModelConfig:
    """Capa 0: conv corta. Capa 1: `mixer`. Zoology alterna los configs de Hybrid por capa."""
    return ModelConfig(
        name=name,
        d_model=d_model,
        n_layers=2,
        vocab_size=vocab_size,
        max_position_embeddings=0,
        block_type="TransformerBlock",
        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer(input_seq_len), mixer]},
        ),
        **extra,
    )


def attention(d_model: int, input_seq_len: int, vocab_size: int = VOCAB_SIZE) -> ModelConfig:
    mixer = dict(name="zoology.mixers.attention.MHA", kwargs={"dropout": 0.1, "num_heads": 2})
    return two_layer_model("attention", d_model, mixer, input_seq_len, vocab_size)


def sliding_window(d_model: int, window: int, input_seq_len: int, vocab_size: int = VOCAB_SIZE) -> ModelConfig:
    mixer = dict(name="zoology.mixers.slide_attn.SlidingAttn", kwargs={"block_size": window, "attention_dropout": 0.0})
    return two_layer_model(f"sliding_window-w{window}", d_model, mixer, input_seq_len, vocab_size)


def gated_delta_net(d_model: int, input_seq_len: int, vocab_size: int = VOCAB_SIZE) -> ModelConfig:
    # Mismos kwargs que add_gated_delta_net en zoology/experiments/models_repo.py
    mixer = dict(
        name="zoology.mixers.gated_delta_net.GatedDeltaNet",
        kwargs={"l_max": input_seq_len, "num_heads": 2, "use_gate": False, "use_short_conv": True, "conv_size": 4},
    )
    return two_layer_model("gated_delta_net", d_model, mixer, input_seq_len, vocab_size)


def gated_delta_net_neg(d_model: int, input_seq_len: int, vocab_size: int = VOCAB_SIZE) -> ModelConfig:
    """Gated DeltaNet con autovalores negativos permitidos (beta en (0,2)).

    Según "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues"
    (ICLR 2025), este cambio de una línea es lo que permite a DeltaNet resolver
    paridad y otras tareas de seguimiento de estado. Lo incluimos como variante
    para ver la diferencia en el banco.
    """
    mixer = dict(
        name="zoology.mixers.gated_delta_net.GatedDeltaNet",
        kwargs={"l_max": input_seq_len, "num_heads": 2, "use_gate": False, "use_short_conv": True, "conv_size": 4,
                "allow_neg_eigval": True},
    )
    return two_layer_model("gated_delta_net_neg", d_model, mixer, input_seq_len, vocab_size)
