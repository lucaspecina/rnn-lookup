# Bitácora

Qué hicimos, qué encontramos y qué decidimos, con fecha. Lo más nuevo arriba.
Cada entrada: qué pasó, por qué, y qué cambia. La explicación general del
proyecto está en `guia.md`; los detalles crudos, en `notas/`.

## 2026-10-02 (noche): resultados del banco v1

**Terminó el banco v1**: 120 corridas, 0 fallas, 1 h 53 min de pared en la VM
(~$5). Tablero: https://claude.ai/artifact/Rz9CzaL6CPzr7HxNjWJptT. Tabla
completa en `results/bench_v1/scorecard.md` (ignorado por git; se regenera).

**El banco está calibrado: muestra las diferencias conocidas, y una sorpresa.**

1. *Recuerdo exacto (MQAR).* Atención 100% en todo, en una época, incluso con
   secuencias 4× más largas que las de entrenamiento. Gated DeltaNet se cae
   donde se le llena la hoja: con estado de 17 KB (d64) perfecto hasta 32 pares,
   45% con 128 y 13% con 256; con 67 KB (d128) aguanta hasta 128 y da 92% con
   256; con 265 KB (d256), 96%. La ventana de 64 se cae apenas los pares no le
   entran (57% con 32, 0 después). Exactamente la curva de Zoology.
2. *Actualización (la clave cambia de valor, vale el último).* **La atención se
   queda en 75-78% en todas las dificultades y Gated DeltaNet le gana**
   (d256: 100% hasta 64 pares, 91% con 128). El 75% tiene explicación exacta:
   la mitad de las claves se actualizan; esta atención no tiene posiciones
   explícitas (solo la conv corta), así que para una clave actualizada ve dos
   valores y no sabe cuál fue el último: acierta las no actualizadas (50%) y
   adivina en las otras (25%). La RNN pisa lo viejo al escribir. Es parte
   configuración (un Transformer con posiciones lo resolvería) y parte
   mecanismo. Primer juego donde la hoja gana.
3. *Claves compuestas.* Misma foto que MQAR pero más duro para la RNN
   (d128: 84% con 144 pares; d64: 36%). Atención 100% en 2 épocas.
4. *Paridad acumulada.* Atención en el azar (50%) en todos los largos, como
   predice la teoría. Gated DeltaNet normal también falla (67% en 64, azar en
   256). **La variante con autovalores negativos (beta en (0,2)) da 100% en 64,
   128 y 256, y 97% en 512: ocho veces el largo de entrenamiento.** Es la
   replicación limpia de arXiv:2411.12537 y la primera "perilla de
   arquitectura" que el banco distingue: un cambio de una línea pasa de azar a
   perfecto.
5. *Mayoría acumulada.* Fácil para todos (97-99% en 512) salvo las ventanas,
   que no pueden contar más allá de lo que ven (62-72% en 512).

**Costo real por corrida.** Suma de minutos de entrenamiento de las 40 mejores
corridas: 441. Las de Gated DeltaNet en MQAR son las caras (26-34 min cada una
con 12 corridas compartiendo las GPUs); paridad y mayoría, 1-3 min.

**Lanzados los juegos de abstracción** (`rules` y `arc1d`, 48 corridas) a las
16:27 hora VM. `arc1d` se examina también con las 860 instancias reales de
1D-ARC que entran en 288 tokens.

## 2026-10-02 (tarde): primera sesión en la VM

**Entorno en la VM, listo.** 2×H100 NVL (94 GB cada una), driver 580, torch 2.11
cu128, fla 0.5.2. El disco del sistema estaba al 98% (otros proyectos: anaconda,
caches de HF y pip); se limpiaron solo caches (pip, apt) y el entorno + cache de
datos van al disco temporal de Azure (/mnt, 256 GB, efímero). `setup_vm.sh` lo
regenera en ~5 minutos si la VM se desaloja.

**Dos bugs que había que encontrar antes de medir nada.**
1. `fla` 0.5 quitó el argumento `head_first` que usaba el wrapper de Gated DeltaNet
   de Zoology. Arreglado en el fork.
2. El Triton que trae torch (3.6) da resultados incorrectos en Hopper en la pasada
   hacia atrás de Gated DeltaNet (fla issue #640); fla directamente se niega a
   correr. Se fija `triton>=3.7.1` en `setup_vm.sh`.
Con eso, la prueba de plomería del banco (40 corridas chicas, 5 juegos, 8
arquitecturas) pasó completa: 0 fallas.

**Primeras mediciones de tiempo (MQAR tamaño completo, 1 época, 180K secuencias).**

| Arquitectura | seg/época | accuracy tras 1 época |
|---|---|---|
| atención d128 | 11 | 1.000 (ya resuelve todo, hasta 1024 tokens / 256 pares) |
| ventana w64 d128 | 11 | 0.514 |
| Gated DeltaNet d128 | 92 | 0.934 |
| Gated DeltaNet d256 | 70 | 0.988 |

Gated DeltaNet es ~8× más lento por época que atención a esta escala (kernels
Triton con heads chicas; la GPU está lejos de saturarse por corrida). Por eso se
corren 6 corridas por GPU en paralelo.

**Banco v1 lanzado** a las 14:33 en tmux (`results/bench_v1/launch.log`): 120
corridas, 12 en paralelo. Estimación previa: 2 a 3 horas, ~$8.

**Detalle de serialización.** Pydantic serializa los segmentos de datos como la
clase base y pierde `num_kv_pairs` y compañía; el `config.json` de cada corrida
ahora se guarda con `serialize_as_any=True` para ser reproducible.

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
