# Recurrent state + lookback: brief para arrancar

Este documento resume una conversación sobre una idea de arquitectura. Ninguna parte es un plan cerrado. Es el contexto y una lista de cosas que estaría bueno explorar. La idea es encontrarle la vuelta juntos.

## La idea en una frase

Un modelo que lleva **un estado latente de tamaño fijo** (o varios), como una RNN moderna, y que además puede **volver atrás en el contexto solo cuando lo necesita**. Volver atrás sería como una acción o "tool", en lugar de mirar todo en cada token como hace la atención.

## Por qué tiene sentido (primeros principios)

- Una RNN comprime todo lo que leyó en un estado fijo. El problema de fondo es que **cuando comprime no sabe qué le van a preguntar después**, así que tira información que más tarde resulta clave.
- Esto está probado formalmente. *RNNs are not Transformers (Yet)* (Wen, Dang, Lyu, ICLR 2025, arXiv:2402.18510) muestra que las RNN no pueden hacer recuperación exacta en contexto (associative recall, etc.). También muestra que **alcanza con agregarles RAG o una sola capa de atención** para cerrar la brecha con los Transformers.
- Conclusión: estado comprimido + capacidad de ir a buscar es, en principio, suficiente. La pregunta es cómo hacerlo barato y aprendible.

## Contexto técnico: qué es lo "moderno"

- **Linear attention**: si sacás el softmax, en vez de guardar todas las keys/values las vas acumulando en una matriz `S += v kᵀ` y leés con `S q`. Es una RNN con estado matricial: un diccionario comprimido clave→valor. El problema es que solo suma y nunca borra, así que las claves se pisan.
- **DeltaNet**: antes de escribir, lee qué dice hoy la memoria para esa clave y escribe solo el error: `S += β (v − S k) kᵀ`. Es un paso de gradiente online sobre `‖S k − v‖²`. El estado es un mini-modelo que se entrena mientras lee (las *fast weights* de Schmidhuber, 1992).
- **Gated DeltaNet** (arXiv:2412.06464): agrega un factor de olvido dependiente del input, `S = α S + ...`. Mamba, KDA (Kimi), Titans y ATLAS son variaciones de la misma familia: memoria fija, entrenada online, con olvido controlado.
- La capacidad sigue siendo finita: una matriz `d×d` guarda limpio del orden de `d` pares. Por eso los modelos en producción son **híbridos**. Qwen3-Next, Qwen3.5 y Kimi Linear usan un patrón 3:1, es decir 3 capas recurrentes por cada capa de atención completa.

## Trabajo relacionado (el espectro)

Va desde el "volver atrás" suave, aprendido dentro del modelo, hasta el discreto, hecho afuera como agente.

| Enfoque | Qué hace | Referencia |
|---|---|---|
| Híbridos 3:1 | Algunas capas siguen con atención completa siempre | Qwen3-Next; Kimi Linear (arXiv:2510.26692) |
| Sparse attention aprendida | Un indexer liviano elige top-k tokens (2048) por query | DeepSeek V3.2 (DSA); NSA |
| RNN + acceso por chunks | Mamba + atención jerárquica sparse sobre chunks top-k; KV offloadeable, memoria casi constante | RAMba / HSA (arXiv:2504.16795) |
| Atención sobre estados comprimidos | Atención sobre snapshots del estado recurrente por chunk | DART (arXiv:2608.02032) |
| Decidir *cuándo* mirar | Un router por token decide si disparar atención global; incluye un baseline de disparo al azar | L2A (arXiv:2603.17484) |
| Memoria que aprende en test time | Memoria neuronal larga, actualizada por "sorpresa" (gradiente) | Titans (arXiv:2501.00663), ATLAS (arXiv:2505.23735) |
| Memorias múltiples / crecientes | Estados a varias escalas; memoria exacta para lo que el estado olvida | Log-linear attention; "A hippocampus for linear attention" (arXiv:2607.02303); "Memory Caching: RNNs with Growing Memory" (2026) |
| Agentes con memoria fija (discreto) | Memoria en tokens reescrita por chunk, entrenada con RL porque no es diferenciable | MemAgent (arXiv:2507.02259), MEM1 |
| Contexto como entorno | El LLM manipula el contexto con código en un REPL y se llama recursivamente | Recursive Language Models (arXiv:2512.24601) |
| Survey reciente agentic | Gestión de contexto disparada por el modelo | Context Language Models (arXiv:2609.37725) |

No encontramos (todavía) algo que haga exactamente esto: **estado latente recurrente + acción de lookback (idealmente discreta) entrenada a nivel arquitectura**. Las piezas existen por separado: L2A resuelve el *cuándo* de forma suave, RAMba el *dónde*, MemAgent y RLM lo discreto pero en espacio de tokens. Conviene seguir revisando la literatura a medida que avancemos, porque el área se mueve rápido.

## Framing honesto: qué se gana y qué no

- **En calidad, casi seguro no se le gana a la atención completa.** Lo mejor posible es igualarla. La apuesta es de **eficiencia**: igualar la calidad de un híbrido gastando mucho menos.
- Dónde está el costo hoy: a contexto largo, generar está limitado por *leer* el KV cache en cada token. Un híbrido 3:1 igual lee todo el KV en el 25% de las capas. DSA reduce la cuenta, pero guarda todo en GPU y el indexer recorre todas las keys.
- Qué podría ganar esta idea:
  - La mayoría de los tokens no miran atrás (costo por token ≈ constante).
  - El pasado se consulta poco, así que puede vivir en memoria barata (CPU, disco).
  - Buscar en el contexto o en una base externa puede ser la misma operación.
- Ojo: **volver atrás no elimina guardar el pasado**. La memoria total sigue siendo O(n); lo que cambia es dónde vive y cuánto se lee. Reencuadre útil: el estado recurrente es "lo que entendí + un índice de dónde está cada cosa".
- La tensión central es discreto vs. suave. Lo discreto da interpretabilidad y largo arbitrario, pero pierde el gradiente (RL, ruidoso y caro). Lo suave entrena con backprop normal. Hay caminos intermedios para explorar.
- Las ganancias reales aparecen a contextos muy largos (agentes, 1M+ tokens). A 8K no hay mucho que ganar.

## Testbed de juguete

**MQAR** (multi-query associative recall), de Zoology (Arora et al., arXiv:2312.04927; repo `HazyResearch/zoology`). Ejemplo: una secuencia de pares `A 4 B 7 C 1 ...` y más adelante consultas `B → 7`. Es recuperación exacta en contexto en estado puro.

- La dificultad se controla con dos perillas: cantidad de pares y tamaño del estado. Así se ve exactamente dónde se cae la RNN pura.
- Escala: 2 capas, dimensión 64–256, secuencias de 256 a 4K. Cada corrida tarda minutos en una sola GPU.
- Implementaciones de Gated DeltaNet y familia: `fla-org/flash-linear-attention`.
- En MQAR se sabe exactamente *dónde* hace falta mirar atrás: en los tokens de consulta. Eso permite verificar a ojo si el mecanismo aprendió algo con sentido.

Siguiente escalón, si lo de juguete promete: un LM chico (~100M parámetros, horas de GPU) con needle-in-a-haystack, passkey y tareas tipo RULER (multi-hop, variable tracking).

## Cosas que estaría bueno explorar (abiertas)

Esto no es una lista de tareas. Son preguntas y direcciones para ir viendo cuáles son más interesantes.

**Mediciones y comparaciones que probablemente necesitemos**
- Baselines naturales: Gated DeltaNet puro (debería caerse), GDN + una capa de atención completa (techo), GDN + sliding window, y quizás algo tipo top-k por chunks.
- Un **presupuesto de lookback** (% de tokens que pueden mirar atrás) como eje principal. El gráfico interesante es accuracy vs. presupuesto.
- Un **control con disparo al azar** con el mismo presupuesto, para separar "aprendió *cuándo* mirar" de "cualquier mirada sirve".
- Generalización de longitud: entrenar corto y testear largo.
- Barrer el tamaño del estado para ver cómo se mueve la frontera.

**Preguntas de diseño donde puede haber cosas interesantes**
- *¿Qué dispara el lookback?* Puede ser un router aprendido. Pero hay algo que nos parece prometedor: la delta rule ya calcula un error, `‖v − S k‖`, que es una señal natural de "mi memoria no sabe esto". ¿Sirve como disparador gratis? También se puede pensar en incertidumbre o entropía de la predicción, o en la "sorpresa" estilo Titans.
- *¿A qué se mira?* Opciones: tokens crudos, chunks, resúmenes de chunks o snapshots del estado recurrente (como DART). Y está la pregunta de qué "migas" deja el modelo al escribir para poder encontrarlas después.
- *¿Qué se hace con lo recuperado?* Se puede sumar a la salida, o **reescribirlo en el estado** (re-consolidar la memoria). Lo segundo podría hacer que no haga falta volver a buscar lo mismo.
- *¿Uno o varios estados?* Memorias a distintas escalas de tiempo, o una fija más una que crece.
- *¿Cómo se entrena algo discreto?* Relajaciones (Gumbel, straight-through, top-k diferenciable), distilación desde un modelo con atención (como el indexer de DSA), entrenar suave y endurecer después, o RL con penalización por presupuesto.
- *¿Qué aprende realmente el router?* Inspeccionar dónde se dispara. En MQAR debería concentrarse en las consultas.

## Cómo nos gusta trabajar

- **Medir antes de optimizar.** Primero reproducir la falla conocida (la curva donde GDN puro se cae en MQAR) y tener los baselines sanos. Recién después tocar la idea.
- Empezar chico y rápido: iteraciones de minutos, no de horas.
- Python. Código simple y legible antes que optimizado.
- Ser crítico con los resultados. Si algo no funciona o la idea no se sostiene, decirlo claro. Un resultado negativo bien medido también sirve.
- Ir anotando hallazgos y decisiones en el repo a medida que aparecen.
