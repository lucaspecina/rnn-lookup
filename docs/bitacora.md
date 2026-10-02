# Bitácora

Qué hicimos, qué encontramos y qué decidimos, con fecha. Lo más nuevo arriba.
Cada entrada: qué pasó, por qué, y qué cambia. La explicación general del
proyecto está en `guia.md`; los detalles crudos, en `notas/`.

## 2026-10-02

**De un experimento a un banco de pruebas.** Lucas marcó que MQAR solo mide lo
que la RNN hace peor, y que lo que buscamos es una búsqueda de arquitecturas:
hace falta medir varias cosas a la vez. El banco v1 (`experiments/bench_v1.py`)
corre cada arquitectura en cinco juegos de Zoology (mqar, forgetting_mqar,
compositional_mqar, cumulative_parity, cumulative_majority), con examen en
secuencias más largas que las de entrenamiento, y `rnn_lookup.scorecard` arma la
tarjeta. 8 arquitecturas x 3 lr x 5 juegos = 120 corridas. Validado en la Mac que
todas las configs construyen datos. MAD (ICML 2024) queda anotado como fuente de
juegos para la v2 (copia selectiva, compresión).

**Variante con autovalores negativos.** Zoology calcula beta con sigmoid, o sea
autovalores en [0,1]: según arXiv:2411.12537 eso impide resolver paridad. Se
agregó `allow_neg_eigval` al wrapper de GatedDeltaNet en el fork (beta en (0,2))
y la variante `gated_delta_net_neg` entra al banco. Es la primera "perilla de
arquitectura" que el banco debería distinguir.

**Repo público: sin nombres internos.** Los identificadores de Azure (grupo de
recursos, VM, usuario) van en `scripts/vm.env`, ignorado por git. Verificado que
no hay credenciales, IDs ni IPs en lo versionado.

**iCloud rompe entornos de Python.** La carpeta Desktop se sincroniza con iCloud
y duplicaba archivos del `.venv` dentro del repo. El entorno local vive ahora en
`~/.venvs/rnn-lookup`, enlazado como `.venv`.

**Reorganización de docs.** Dos documentos pulidos para Lucas (`guia.md`, que
explica desde cero, y esta bitácora) y el resto en `notas/`. Regla: no crear docs
nuevos en `docs/` sin preguntar.

## 2026-10-01

**Pregunta central del proyecto.** Con el mismo libro guardado afuera y el mismo
presupuesto de consultas, ¿sirve de algo tener una cabeza que *resume* (estado
recurrente aprendido) en lugar de una que *recorta* (ventana de los últimos N
tokens)? Rivales: (1) Transformer completo como techo de referencia, (2) ventana
+ consultar, el parche de hoy, (3) RNN + consultar, nuestra idea. Gráfico
objetivo: accuracy vs. presupuesto de consultas, una curva por cabeza. Si las
curvas se pegan, el resumen aprendido no aporta y la idea muere ahí.

**Escalera de evaluación.** Nivel 1 (MQAR, 2 capas, minutos) en la VM; lo que
promete pasa a nivel 2 (LM de 124M a 340M en FineWeb-Edu, recall-intensive y
S-NIAH). Detalle y fuentes en `notas/como-evaluan.md`.

**Hardware.** La Mac (M1 Pro, 16 GB) solo para escribir código. Experimentos en
una VM spot con 2×H100 (~$2.58/h) que ya existía en la suscripción de Azure. Alternativa más barata si alcanza una GPU: spot 1×H100 a ~$1.29/h.
Las compute instances de Azure ML (1×H100 on-demand) salen $6.98/h: no usarlas para esto.
Cuotas sin pedir aumento: solo H100 en la región principal y una T4 en otra.

**Zoology como fork + submódulo.** Zoology trae MQAR, el entrenamiento, los
sweeps y los baselines (atención, ventana, Gated DeltaNet). Pero solo guarda
resultados en wandb; sin wandb, imprime y descarta. Necesitábamos resultados en
archivos, así que: fork `lucaspecina/zoology`, rama `rnn-lookup`, con un logger
local en JSONL. Submódulo en `third_party/zoology` para que el diff contra
upstream quede visible y chico. Los mixers nuestros van en `rnn_lookup/`, porque
Zoology carga módulos por nombre de import.

**Sin causal_conv1d ni mamba_ssm.** Son las dependencias problemáticas de
Zoology. No hacen falta: la conv corta de `fla` usa su backend Triton por
defecto. Se instala el fork con `--no-deps` y las dependencias a mano.

**Primer experimento.** Reproducir la caída de GDN puro en MQAR con la receta
de datos del sweep original de Zoology. (Al día siguiente se amplió al banco de
cinco juegos; ver 2026-10-02.) Pendiente: medir cuánto tarda cada corrida.
