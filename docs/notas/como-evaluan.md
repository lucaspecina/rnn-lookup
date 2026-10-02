# Cómo evalúan de verdad los papers de esta familia

Relevado el 2026-10-01 a partir de los papers y repos listados al final. Todo lo
que está acá sale de ahí; lo que no encontramos está marcado como tal.

La práctica real es una escalera de tres niveles. Cada nivel es más caro y más
realista que el anterior, y nadie salta al siguiente sin resultados en el previo.

## Nivel 1: tareas sintéticas (minutos, 1 GPU)

**MQAR** (multi-query associative recall), definido en Zoology (Arora et al.,
arXiv:2312.04927). La secuencia tiene pares clave→valor y después consultas:

```
A 4  B 3  C 6  F 1  E 2   →   A ?  C ?  F ?  E ?  B ?
```

El modelo tiene que responder cada consulta con el valor que vino después de esa
clave. Vocabulario de 8.192 tokens.

Configuración real del sweep del repo `HazyResearch/zoology`
(`zoology/experiments/mqar_example_configs/original_mqar_configs.py`):

| Perilla | Valores |
|---|---|
| Largo de secuencia | 64, 128, 256, 512, 1024 |
| Pares clave→valor | 4, 8, 16, 32, 64, 128, 256 |
| Capas | 2 |
| Learning rate | 4 valores en `logspace(-3, -1.5)` |
| Ejemplos de entrenamiento | 100.000 (20.000 para los largos) |
| Ejemplos de test | 1.000 |
| Épocas | 32 |
| Batch | 256 |
| Mixers comparados | attention, sliding_window, gated_delta_net, ttt_linear, ttt_mlp |

Se reporta la **máxima accuracy de test** sobre el barrido de learning rates.
Los mixers ya incluidos (attention, sliding_window, gated_delta_net) son
exactamente nuestros baselines.

**Variante de Based** (arXiv:2402.18668): entrenan con secuencias de 256 tokens
y 4 a 64 pares, y evalúan con 1.024 tokens y 4 a 256 pares. O sea, miden
generalización a largos no vistos. Su gráfico central es **tamaño del estado
(bytes) vs. accuracy en MQAR**, una curva por arquitectura. Es el molde de
nuestro gráfico "presupuesto de consultas vs. accuracy".

**Por qué el sintético vale.** Zoology muestra que en el Pile los tokens que
requieren recuperación exacta ("AR hits") son el 6,4% del total pero explican el
82% de la brecha de perplexity entre atención y las alternativas. Un Transformer
de 70M predice esos tokens mejor que un Hyena de 1.4B. Por eso MQAR predice lo
que después pasa en texto real.

**Lo que no encontramos:** ningún paper reporta cuánto tarda una corrida
sintética ni en qué GPU. Hay que medirlo nosotros.

## Nivel 2: modelo de lenguaje chico (horas a días, varias GPUs)

Escalas usadas:

| Paper | Tamaños | Tokens | Datos |
|---|---|---|---|
| Based | 360M, 1.3B | 10B a 50B | Pile |
| Gated DeltaNet (arXiv:2412.06464) | 400M, 1.3B | 100B | FineWeb-Edu |

Receta de Gated DeltaNet: secuencia 4K, batch de 0,5M tokens, AdamW con lr 4e-4,
weight decay 0,1, cosine con 1B tokens de warm-up, tokenizer de Llama2 (32K).

Framework: `fla-org/flame` (sobre torchtitan), pensado para entrenar modelos de
`flash-linear-attention` en FineWeb-Edu. El ejemplo del README es un Transformer
de 340M, 8 GPUs, 20.480 pasos (unos 10B tokens, calculado por nosotros a partir
de los parámetros del ejemplo).

Para iterar rápido en texto real existe el **speedrun de modded-nanogpt**:
124M parámetros en FineWeb hasta val loss 3,28; récord de 0,665 minutos en
8×H100 con menos de 330M tokens (agosto 2026). No es contexto largo, pero muestra
cuán corto puede ser el loop si el setup está bien armado.

Evaluación en este nivel:

- **Lenguaje general (zero-shot, lm-eval-harness):** PIQA, HellaSwag, WinoGrande,
  ARC-easy, ARC-challenge, SIQA, BoolQ; perplexity en Wikipedia y LAMBADA.
- **Recall-intensive** (introducido por Based, adoptado por Gated DeltaNet):
  SWDE, FDA, SQuAD, NQ, TriviaQA, DROP, con el input truncado a 2K tokens. Esto
  es lo que separa "sabe lenguaje" de "recupera un dato exacto del contexto".
  Es la métrica que más nos importa.

## Nivel 3: contexto largo

- **S-NIAH** (de RULER, arXiv:2404.06654): una aguja (passkey, número o UUID) en
  un pajar de 1K, 2K, 4K y 8K tokens. Gated DeltaNet lo reporta a 1.3B.
- **RULER completo:** variantes de NIAH con varias claves/valores/consultas,
  variable tracking (seguir cadenas de asignaciones), agregación (palabras más
  frecuentes) y QA con distractores.
- **LongBench:** 14 tareas (NarrativeQA, Qasper, HotpotQA, GovReport, etc.).
- **Extrapolación de largo:** perplexity hasta 20K tokens en PG19, CodeParrot,
  GovReport y otros.

## Qué hacen los labs grandes (Kimi, MiniMax, Qwen)

Lo que publican sobre cómo decidieron la arquitectura, no sobre el modelo final.

**Kimi Linear** (Moonshot, arXiv:2510.26692):

- *Sintético:* modelos de 2 capas, 2 heads, head dim 128. Tareas: palindrome,
  MQAR y stack. Comparan KDA vs. Gated DeltaNet vs. Mamba2 con secuencias de 256
  a 2.048 tokens, hasta 20.000 pasos.
- *Escalera de scaling:* 5 tamaños (653M a 1.7B parámetros activos, MoE), hasta
  102,5B tokens. Gráfico: loss vs. cómputo (PFLOPs-día). Comparan KDA vs. MLA.
  Resultado: ~1,16× de eficiencia de cómputo.
- *Ablación del ratio híbrido:* a escala real (48B totales / 3B activos, 1,4T
  tokens). Ratios 0:1, 1:1, 3:1, 7:1 y 15:1. Métrica: perplexity de validación.
  Gana 3:1. Benchmarks: MMLU-Pro, BBH, HellaSwag, GSM8K, C-Eval, CMMLU y otros.
- *Largo:* RULER a 128K, MRCR, HELMET, LongBench v2, RepoQA, Long Code Arena.
- *Eficiencia:* tiempo por token de salida 6× mejor que MLA a 1M de contexto,
  KV cache hasta 75% menor.

**MiniMax-01** (arXiv:2501.08313):

- *Escalera de scaling:* 6 tamaños (70M, 160M, 410M, 1B, 3B, 7B), hasta 300B
  tokens, secuencia 8.192. Comparan atención softmax, lightning attention pura y
  el híbrido (1 capa softmax cada 8). Ajustan una ley de potencia loss vs.
  cómputo.
- *Downstream en esa escalera:* BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC,
  OpenBookQA, NIAH y SCROLLS.
- *Hallazgo clave:* la atención lineal pura iguala a softmax en todo **salvo
  NIAH** (aguja en el pajar). Eso es lo que justifica el híbrido.
- *Cómo eligieron 7:1:* no está documentado.

**Qwen3-Next** (blog, septiembre 2025) y **Qwen3.8-Next** (arXiv:2608.30320):

- El blog dice que Gated DeltaNet le ganó a sliding window y a Mamba2 en
  in-context learning, y que 3:1 fue el mejor ratio. El protocolo no se publicó.
- El paper de diseño de Qwen3.8-Next evalúa cada cambio en tres ejes: loss más
  14 benchmarks de pre-entrenamiento; costo en training, prefill y decode; y
  efecto sobre hiperparámetros y estabilidad. Advertencia textual: "loss y
  accuracy downstream no siempre se mueven juntas".

**El patrón común:**

1. El ancla es la **escalera de scaling**: 5 o 6 tamaños, mismos datos, loss
   vs. cómputo. La arquitectura nueva tiene que quedar por debajo en todos los
   tamaños. Si no, se descarta aunque mejore otra cosa.
2. Lo específico del objetivo va aparte. Para memoria: sintético de 2 capas
   (MQAR y parientes) más aguja en el pajar o RULER a largos grandes.
3. Las ablaciones caras (ratios, variantes de gating) se hacen a escala real y
   solo los labs pueden pagarlas.
4. La eficiencia se mide por separado: throughput y memoria de 4K a 1M.

## Repos que ya tienen esto armado

Verificado el 2026-10-01 mirando los repos.

| Nivel | Repo | Qué trae | Corre en la Mac |
|---|---|---|---|
| 1 | `HazyResearch/zoology` | Generador de MQAR, harness de entrenamiento, sweeps, logging en wandb. Mixers ya implementados: `attention.py`, `slide_attn.py`, `delta_net.py`, `gated_delta_net.py`, `mamba2.py`, `gla.py`, `based.py`, `hybrid.py`, `deepseek_nsa.py` y más. | A medias. El harness y la atención sí. `gated_delta_net.py` importa `fla.ops.gated_delta_rule` (kernels Triton, solo CUDA). |
| 1 | `fla-org/flash-linear-attention`, archivo `fla/ops/gated_delta_rule/naive.py` | Dos implementaciones de Gated DeltaNet en PyTorch puro, sin Triton: `naive_recurrent_gated_delta_rule` (loop token a token) y `naive_chunk_gated_delta_rule` (por bloques de 64). | Sí. Es la pieza que nos permite correr GDN acá. |
| 2 | `fla-org/flame` | Entrenamiento de modelos `fla` en FineWeb-Edu sobre torchtitan. Configs de 340M y 1.3B. | No (CUDA, varias GPUs). |
| 2 | `KellerJordan/modded-nanogpt` | Loop ultra corto en texto real: 124M en FineWeb. | No (pensado para 8×H100). |
| 2 | `EleutherAI/lm-evaluation-harness` | Los quizzes estándar (PIQA, HellaSwag, ARC, etc.). | Evaluar sí, si el modelo entra. |
| 2 | `HazyResearch/based`, carpeta `evaluate` | Las tareas de recuperación exacta: `swde`, `fda`, `squad_completion`. Comando: `python launch.py --task swde --task fda --task squad_completion --model ...` | Evaluar sí, si el modelo entra. |
| 3 | `NVIDIA/RULER` | Aguja en el pajar y variantes, variable tracking, agregación, a largos configurables. | Evaluar sí, si el modelo entra. |

Lectura: el nivel 1 está casi resuelto por Zoology, con una sola pieza a
reemplazar (el mixer de Gated DeltaNet, por la versión naive de `fla`). El nivel
2 ya tiene framework, datos y evaluaciones; solo hace falta la GPU.

## Nuestro plan

1. **Nivel 1 en la workstation NVIDIA (Ubuntu) o en una VM.** Zoology tal
   cual, con los kernels de `fla`. La Mac queda para escribir código. Primero
   reproducir la curva donde GDN puro se cae en MQAR y tener los baselines sanos
   (atención, ventana). Recién después sumar el lookback como un mixer más.
   Cada corrida, minutos. La versión naive de GDN en PyTorch puro sigue siendo
   útil como referencia legible cuando toquemos la recurrencia.
2. **Lo que promete pasa al nivel 2 en Azure.** 124M a 340M en FineWeb-Edu con
   `flame`, evaluando recall-intensive (`based/evaluate`) y S-NIAH. Una idea que
   gana en MQAR pero empeora la loss general se descarta.
3. **Nivel 3 solo con un modelo del nivel 2 que haya pasado.**

Regla que vale en todos los niveles: todo fijo, cambia una sola pieza, varios
learning rates y semillas, y se reporta cuánto se sostiene el resultado al subir
de escala.

## Qué significa para nosotros

1. El loop de iteración es el nivel 1, con el sweep de Zoology tal cual, sumando
   nuestro mecanismo como un mixer más. Los baselines ya están en el repo.
2. El gráfico objetivo es el de Based, cambiando el eje x: en vez de bytes de
   estado, presupuesto de consultas.
3. Nivel 2 recién si el nivel 1 promete: 124M a 340M en FineWeb-Edu, evaluando
   recall-intensive y S-NIAH. Ahí entra la GPU en Azure.
4. Falta medir el tiempo real de una corrida sintética en nuestro hardware.

## Fuentes

- Zoology: https://arxiv.org/abs/2312.04927 y https://github.com/HazyResearch/zoology
- Based: https://arxiv.org/abs/2402.18668
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- RULER: https://arxiv.org/abs/2404.06654
- flame: https://github.com/fla-org/flame y
  https://github.com/fla-org/flash-linear-attention/blob/main/examples/training.md
- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt
- Based (repo con las evaluaciones de recall): https://github.com/HazyResearch/based
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- RULER (repo): https://github.com/NVIDIA/RULER
- GDN naive en PyTorch puro: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py
- Kimi Linear: https://arxiv.org/abs/2510.26692
- MiniMax-01: https://arxiv.org/abs/2501.08313
- Qwen3-Next (blog): https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd
- Qwen3.8-Next design paper: https://arxiv.org/abs/2608.30320
