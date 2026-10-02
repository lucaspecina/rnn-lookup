# rnn-lookup

Investigación de arquitectura: un modelo con **estado recurrente de tamaño fijo**
(tipo Gated DeltaNet) que puede **volver a consultar el contexto solo cuando lo
necesita**, en lugar de mirar todo en cada token como la atención. La apuesta es
de eficiencia a contexto largo, no de calidad. Explicación completa desde cero
en `docs/guia.md`. Lucas habla castellano; todo el repo va en castellano.

## Cómo trabajamos

- **Es un banco de pruebas, no un experimento.** Entra una arquitectura, sale una
  tarjeta de puntajes sobre varios juegos (recuerdo, actualización, claves
  compuestas, paridad, mayoría, generalización de largo) más memoria y costo.
  Primero calibrar el banco con los baselines; recién después buscar arquitecturas.
- **Chico y rápido.** Nivel 1 (MQAR, 2 capas, minutos por corrida) hasta que algo
  prometa; recién ahí nivel 2 (LM de 124M a 340M). Escalera en `docs/guia.md` §5.
- **Todo fijo, cambia una sola pieza.** Mismos datos, mismo tamaño, varios
  learning rates y semillas. Se reporta cuánto se sostiene al subir de escala.
- Python simple y legible antes que optimizado. Sin jerga sin definir en los docs.
- Ser crítico: un resultado negativo bien medido también sirve, y se escribe.
- **Documentos: dos pulidos y el resto en notas.** `docs/guia.md` (explica el
  proyecto desde cero, sin jerga sin definir) y `docs/bitacora.md` (qué pasó, con
  fecha) son para Lucas y se mantienen cortos y claros. Todo lo demás (investigación,
  números crudos, fuentes, borradores) va a `docs/notas/`. No crear docs nuevos
  en `docs/` sin preguntar.
- Cada decisión o hallazgo va a `docs/bitacora.md` cuando aparece. Si cambia la
  explicación general, actualizar `docs/guia.md`.
- Flujo de trabajo general (presentar antes de commitear, etc.): skill `dev-workflow`.

## Estructura

```
docs/guia.md             el proyecto explicado desde cero (pulido, para Lucas)
docs/bitacora.md         qué hicimos, qué encontramos, qué decidimos, con fecha
docs/notas/              investigación cruda: brief.md, como-evaluan.md, ...
rnn_lookup/              código propio (helpers de config, mixers nuevos)
experiments/bench_v1.py  el banco: 5 juegos x 8 arquitecturas x 3 lr (ver docstring)
experiments/mqar/smoke.py prueba de humo de la plomería
rnn_lookup/scorecard.py  arma la tarjeta de puntajes desde results/
scripts/vm.sh            encender/apagar/entrar a la VM con GPU (desde la Mac)
scripts/setup_vm.sh      instalar el entorno (en la VM)
third_party/zoology/     submódulo: nuestro fork de HazyResearch/zoology
results/, data/          salidas y cache de datos (ignorados por git)
```

## Zoology (submódulo)

- Fork en `lucaspecina/zoology`, rama `rnn-lookup`, remoto `upstream` apunta al
  original. Cambios nuestros: logger local en JSONL (`TrainConfig.output_dir`),
  wandb opcional, `ZOOLOGY_WORKERS_PER_GPU`. Mantener el diff contra upstream chico.
- Cambios en el fork hasta ahora: logger local, workers por GPU, CPU sin CUDA,
  `allow_neg_eigval` en GatedDeltaNet. Los mixers nuevos van en `rnn_lookup/`, no
  en el fork: Zoology los carga por nombre de import (`ModuleConfig(name="rnn_lookup.mixers....")`).
- Cada corrida deja `results/mqar/<sweep>/<run>/{config.json, model.json, metrics.jsonl}`.
  La métrica que importa es `valid/num_kv_pairs/accuracy-<k>` (máximo sobre épocas).

## Comandos

```bash
scripts/vm.sh status | start | stop | ssh | sync | run CMD | fetch   # la VM cobra solo encendida
bash scripts/setup_vm.sh                                             # en la VM, una vez
python -m zoology.launch experiments/mqar/smoke.py                   # prueba de humo
BENCH_SMOKE=1 python -m zoology.launch experiments/bench_v1.py                   # plomería de los 5 juegos
ZOOLOGY_WORKERS_PER_GPU=4 python -m zoology.launch experiments/bench_v1.py -p   # banco completo
python -m rnn_lookup.scorecard results/bench_v1                                 # tarjeta de puntajes
```

Localmente (Mac, sin GPU) hay un `.venv` con torch CPU que alcanza para la prueba
de humo con atención. Gated DeltaNet necesita los kernels Triton: solo en la VM.

**Apagar la VM al terminar** (`scripts/vm.sh stop`). Es spot: puede apagarse sola;
los resultados viven en `results/` y se traen con `scripts/vm.sh fetch`.
