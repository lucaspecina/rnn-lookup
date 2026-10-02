# Mini-ARC: un juego de abstracción para el banco (borrador)

Nota de investigación, 2026-10-02. Pregunta de Lucas: ¿cómo armar un juego chico,
con datos generados, que mida abstracción reutilizable, al estilo ARC-AGI?

## La esencia de ARC en una secuencia

Cada secuencia trae una regla **nueva**, K ejemplos de entrada→salida y una
entrada de consulta. El modelo produce la salida token a token. La regla cambia
en cada secuencia: no se puede memorizar en los pesos, hay que inducirla de los
ejemplos (aprendizaje en contexto).

```
[3 1 4 1] → [1 4 1 3]  ;  [2 2 5] → [2 5 2]  ;  [7 0 6 6] → ?     (regla: rotar 1)
```

## Diseño propuesto (v0)

- **Primitivas** sobre tiras de símbolos: invertir, rotar k, sustituir a→b,
  borrar símbolo, duplicar, ordenar, aplicar solo después de una marca.
  Inspiradas en 1D-ARC (abajo): mover, rellenar, vaciar, copiar patrón,
  espejar, recolorear por tamaño o por paridad.
- **Reglas** = composición de 1 a 3 primitivas. El generador sortea la regla,
  los K ejemplos y la consulta; etiquetas solo en los tokens de la respuesta.
- **Perillas**: K (1 a 5), largo de las tiras, profundidad de composición,
  vocabulario.
- **Slices de examen** (lo que mide abstracción):
  1. composiciones vistas (control);
  2. composiciones **no vistas** de primitivas conocidas (¿piezas reutilizables?);
  3. primitivas **no vistas** (¿induce una regla nueva solo de los ejemplos?).
- **Protocolo de olvido**: entrenar con el conjunto de primitivas A, después con
  B, volver a medir A.
- Métricas: precisión por token de respuesta y acierto exacto por consulta.

Conexión con la idea del proyecto: la hoja (estado) debería quedarse con la
regla inducida; el libro (contexto) guarda los ejemplos. Un modelo que abstrae
no necesita volver a mirar los ejemplos; uno que no abstrae, sí.

## Antecedentes (verificados por encima, no leídos a fondo)

- **1D-ARC** (Xu, Li, Vaezipoor, Sanner, Khalil, 2023, arXiv:2305.18354): versión
  en una dimensión de ARC, mismos priors de conocimiento básico, 18 tipos de
  tarea con 50 instancias cada uno: Move 1, Move Dynamic, Scaling, Fill, Hollow,
  Pattern Copy, Flip, Mirror, Recolour by Odd/Even, Recolour by Size, etc.
  Pensado para evaluar LLMs, no para entrenar; nos sirve como catálogo de
  primitivas con "objetos" (tramos de píxeles del mismo color).
- **"When can transformers compositionally generalize in-context?"**
  (arXiv:2407.12275, workshop ICML 2024): justo el protocolo de dejar afuera
  combinaciones de módulos en el entrenamiento. Hallazgo: los Transformers
  entrenados en contexto **no generalizan composicionalmente** aunque podrían
  expresarlo. O sea, acá hay un juego donde la arquitectura importa de verdad.
- **"In-Context Compositional Learning via Sparse Coding Transformer"**
  (NeurIPS 2025, arXiv:2511.20194): modifica la atención para codificar reglas
  composicionales; evalúa en S-RAVEN/RAVEN.
- Relacionados: SCAN (generalización composicional), "Learning to Execute",
  aprendizaje en contexto de funciones (Garg et al. 2022).

## Revisión de 1D-ARC (2026-10-02, mirando el repo `khalil-research/1D-ARC`)

Formato: JSON al estilo ARC, con pares train/test. Cada "imagen" es una fila de
píxeles con colores 0 a 9; el 0 es fondo. Los **objetos** son tramos contiguos
de un mismo color. Cada tipo de tarea tiene ~50 instancias con 2 o 3 pares de
ejemplo y 1 de prueba. No hay generador: es un dataset cerrado (MIT).

Los 18 tipos, con ejemplos reales:

| Tipo | Qué hace | Ejemplo (entrada → salida) |
|---|---|---|
| move_1p / 2p / 3p | mover el objeto k píxeles a la derecha | `0006660000` → `0000666000` |
| move_dp, move_2p_dp | mover una distancia que depende de algo de la entrada | |
| fill | rellenar entre dos píxeles del mismo color | `0007000000070000` → `0007777777770000` |
| padded_fill | igual, dejando bordes | |
| hollow | vaciar el interior de un objeto, dejar los extremos | `0007777777770000` → `0007000000070000` |
| flip, mirror | voltear / espejar | |
| denoising_1c / mc | borrar píxeles sueltos (ruido), uno o varios colores | |
| pcopy_1c / mc | copiar un patrón en cada marcador | `03330000300000300030` → `03330003330003330333` |
| recolor_cnt | recolorear cada objeto según su tamaño (1→1, 2→8, 3→5) | `0200022000222000220` → `0100088000555000880` |
| recolor_oe | recolorear según posición par/impar | |
| recolor_cmp | recolorear comparando tamaños entre objetos | |
| scale_dp | estirar el objeto hasta un marcador | `2222222222222222000300000` → `2222222222222222222300000` |

**Lo que cambia para nuestro diseño.** Mi v0 trabaja sobre tiras densas de
símbolos sin fondo. 1D-ARC muestra que lo esencial de ARC está en otro lado:
hay un **fondo** y hay **objetos** (tramos), y las reglas hablan de objetos
(moverlos, rellenarlos, contarlos, compararlos, copiarlos hasta un marcador).
Eso es lo que obliga a percibir estructura antes de aplicar una regla. La v1
del juego debería ser así: filas con fondo 0 y objetos, y primitivas tomadas
de este catálogo (mover k, rellenar, vaciar, recolorear por tamaño con un
mapa de colores sorteado por secuencia, copiar patrón, estirar hasta marcador,
quitar ruido, voltear, espejar), componibles de a 1 a 3, con las mismas slices
de examen (composiciones no vistas, primitivas no vistas) y el protocolo de
olvido. Las 10 primitivas de la v0 quedan como el caso "sin objetos", útil
para comparar.

## Siguiente paso

1. La v0 (`rnn_lookup/games/rules.py`, tiras sin fondo) ya está en el banco
   como juego `rules` y corre después del banco v1 (24 corridas).
2. La v1 (`arc1d`): generador con fondo y objetos siguiendo el catálogo de
   1D-ARC, mismas slices y protocolo de olvido. Entra con el banco v2 junto
   con las tareas de MAD (copia selectiva, compresión).
