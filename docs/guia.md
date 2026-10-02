# Guía del proyecto, desde cero

Este es el documento para entender el proyecto entero sin haber leído nada
antes. Se lee en unos 20 minutos. Cada término técnico se define la primera vez
que aparece. Los detalles finos, las fuentes y los números crudos están en
`docs/notas/`; acá está lo que hace falta para entender y decidir.

El documento hermano es `docs/bitacora.md`: qué hicimos, qué encontramos y qué
decidimos, con fecha. Este explica; aquel registra.

---

## 0. Lo que buscamos de fondo

Antes del problema concreto, el norte. Lo que a Lucas le interesa probar son
sistemas que **abstraigan conceptos** y que esas abstracciones sirvan para
cosas nuevas: estrategias generales para adaptarse y resolver lo que nunca
vieron, en la línea de la filosofía de ARC-AGI. Y que **aprender algo nuevo no
borre lo anterior**, como les pasa a las redes de hoy.

La idea de este proyecto (una hoja de apuntes de tamaño fijo más la capacidad
de ir a consultar el contexto) es un vehículo para eso, con un encuadre que
conviene tener presente: la hoja debería guardar **abstracciones** (reglas,
conceptos) y el libro, los **casos concretos**. Consultar el libro es traer
instancias; lo que queda en la hoja es la regla.

El banco de pruebas de hoy (§6) mide memoria. Dos de sus juegos tocan la
abstracción de refilón: generalizar a secuencias más largas que las de
entrenamiento (aplicar la misma regla a algo nunca visto) y las claves
compuestas. Pero todavía no mide las dos cosas de arriba. Las dos se pueden
armar en miniatura, y van a entrar en la versión siguiente:

1. **Inducir una regla desde pocos ejemplos y aplicarla**: juegos donde el
   "diccionario" cambia en cada secuencia, así que memorizar no sirve y hay que
   abstraer. Zoology ya trae una variante así.
2. **Aprender B sin borrar A**: no es un juego sino un protocolo. Entrenar en A,
   después en B, y volver a medir A.

## 1. El problema

Un modelo de lenguaje lee texto y, más adelante, tiene que usar lo que leyó:
contestar una pregunta sobre el documento, seguir una conversación larga,
acordarse de una instrucción que le dieron hace mil páginas. Para eso tiene que
*recordar*. Y hoy hay dos formas de recordar, con problemas opuestos.

**Guardar todo.** Es lo que hacen los Transformers, la arquitectura de casi
todos los modelos actuales. Imaginate que leés un libro y dejás todas las
páginas abiertas sobre la mesa. Cada vez que vas a escribir una palabra, le
pegás una mirada a todas las páginas. Nunca te olvidás de nada. Pero cuanto más
largo el libro, más páginas hay que mirar por cada palabra: cada palabra nueva
cuesta más que la anterior, y la mesa se llena. Por eso los modelos tienen una
"ventana de contexto": un largo máximo (hoy, entre 200 mil y un millón de
palabras) pasado el cual simplemente no pueden seguir leyendo.

**Tomar apuntes en una sola hoja.** Es lo que hacen las RNN (redes recurrentes).
Leés el libro y vas actualizando una hoja de apuntes de tamaño fijo. Cada
palabra cuesta lo mismo, hayas leído diez páginas o diez mil, y podés leer para
siempre. El problema es que **cuando anotás no sabés qué te van a preguntar
después**. La hoja tiene espacio limitado, así que tirás cosas, y a veces lo que
tiraste era justo el dato que importaba. Esto está demostrado formalmente: una
RNN pura no puede hacer recuperación exacta de algo que leyó, del tipo "¿qué
número venía después de la B?". El mismo trabajo que lo demuestra muestra que
alcanza con agregarle una forma de ir a buscar (una sola capa que mire todo, o
una búsqueda externa) para cerrar la brecha.

Resumiendo el problema: lo que recuerda todo no escala en largo, y lo que escala
en largo no recuerda todo.

## 2. La idea

Es lo que hace una persona. Leés tomando apuntes en la cabeza. La mayoría de las
veces los apuntes alcanzan. Cuando te preguntan un dato puntual que no tenés,
**vas al libro y lo buscás**, en la página que corresponde, no releés todo.

Traducido a arquitectura: un modelo con hoja de apuntes (estado recurrente de
tamaño fijo) que además tiene una acción de **volver a mirar el contexto**, que
se dispara solo cuando los apuntes no alcanzan, y que trae una parte, no todo.

Tres aclaraciones honestas, porque definen qué estamos buscando:

- **No vamos a ganarle en calidad al Transformer.** El que guarda todo acierta
  siempre. Lo mejor posible es igualarlo. La apuesta es de **eficiencia**: misma
  calidad gastando mucho menos, sobre todo en textos muy largos. A 8 mil
  palabras no hay nada que ganar; a un millón, sí.
- **Volver atrás no elimina guardar el pasado.** El libro tiene que existir en
  algún lado. Lo que cambia es dónde vive (puede ser memoria barata, fuera de la
  GPU) y cuántas veces se lee (poco, en lugar de en cada palabra).
- **Todavía no sabemos si funciona.** Las piezas existen por separado en la
  literatura; nadie publicó exactamente esto. Puede que no ande. Un resultado
  negativo bien medido también es un resultado.

## 3. Cómo funciona por dentro

Lo mínimo para seguir las conversaciones y los experimentos. Hay un dibujo de
todo esto en https://claude.ai/artifact/C82HN37tZz9p2tmrZkFmXc.

### Tokens y capas

El texto entra al modelo partido en **tokens** (palabras o pedazos de palabra).
El modelo es una pila de **capas**, digamos 32. Cada token pasa por las 32 capas,
una tras otra, y a la salida de la última el modelo predice el token siguiente.
Lo que diferencia a las arquitecturas es qué hace cada capa para que el token
actual se entere de lo que dijeron los tokens anteriores.

### La capa de atención (la del Transformer)

Cuando el token número t pasa por una capa de atención, la capa calcula tres
vectores a partir de él: un **query** ("qué estoy buscando"), una **key** ("cómo
me pueden encontrar") y un **value** ("qué información aporto"). La key y el
value se guardan. Después, el query del token t se compara contra las keys de
**todos** los tokens anteriores, y la salida es un promedio de sus values, pesado
por cuánto matcheó cada uno.

Eso es "mirar todo", y pasa en cada capa, para **cada token**, de verdad. El
token número 100.000 hace 100.000 comparaciones por capa. Las keys y values
guardados forman el **KV cache**, que crece una entrada por token leído. Ahí está
el costo que queremos evitar.

### La capa recurrente (la hoja de apuntes)

Arranca igual: de cada token saca query, key y value. La diferencia es qué hace
con ellos. En vez de guardar a cada token por separado, tiene un único bloque de
números de tamaño fijo, el **estado**, y cada token que pasa se mezcla adentro.
Para responder, el token actual le pregunta al bloque y listo. No hay lista que
recorrer: cuesta lo mismo con 100 tokens que con un millón.

Un ejemplo en miniatura, con vectores de 2 números (en la realidad son 128 o
más). El estado es una grilla de 2×2 que arranca en cero. Escribir un token es
sumar, en cada celda, `value[fila] × key[columna]`:

```
token 1: key=(1,0) value=(3,5)  →  S = [[3,0],[5,0]]
token 2: key=(0,1) value=(2,7)  →  S = [[3,2],[5,7]]
leer con query=(1,0):  S·q = (3,5)   ✓ recuperó el value del token 1
leer con query=(0,1):  S·q = (2,7)   ✓ recuperó el del token 2
```

Funciona porque las dos keys apuntan a direcciones distintas y no se tocan. El
problema aparece en la palabra "mezcla": con 2 números solo hay 2 direcciones
que no se tocan. Una tercera key, tipo (0.7, 0.7), pisa un poco a las otras dos
al leer. Con vectores de 128 números entran limpias unas 128 cosas; después se
empiezan a confundir. **Eso es lo que la RNN olvida.**

**Gated DeltaNet** es la versión moderna de esta capa, y la que usamos. Mezcla
con más cuidado: antes de escribir, lee qué hay en el estado para esa key y
escribe solo la diferencia (reemplaza en lugar de apilar), y además multiplica
el estado por un factor menor a 1 para ir olvidando lo viejo. Pero el bloque
sigue siendo fijo. La familia entera se llama **linear attention**: atención sin
la operación (el softmax) que obliga a guardar cada token por separado.

### El híbrido, y la diferencia con nuestra idea

Los modelos abiertos de punta (Qwen3-Next, Kimi Linear) son **híbridos**: apilan
las dos capas, típicamente 3 recurrentes por cada 1 de atención. Entre mensajes
guardan las dos cosas, la hoja y el KV cache. Y para cada token nuevo, la capa de
atención mira el KV cache entero, **siempre**, haga falta o no. Más barato que un
Transformer puro (solo un cuarto de las capas mira todo), pero el costo sigue
creciendo con el largo.

Nuestra idea saca esa capa y la reemplaza por una consulta que ocurre solo a
veces. La única diferencia está en una flecha: en el híbrido es sólida y pasa en
cada token; en lo nuestro es punteada, pasa cuando el estado no alcanza, y va a
buscar una parte. Por eso el híbrido es el pariente más cercano de la idea y una
de las comparaciones obligadas.

## 4. Qué hacen hoy los mejores, y contra qué competimos

Todo lo que hace el estado del arte es una respuesta a la misma pregunta: "¿cómo
evito releer el libro entero en cada palabra?". Las respuestas viven en dos
niveles.

**Adentro del modelo (arquitectura).** Los híbridos de arriba. Y la **atención
dispersa** (DeepSeek V3.2): cada token mira solo las ~2000 entradas más
relevantes del KV cache, elegidas por un buscador barato. Gasta menos cómputo,
pero el libro completo sigue viviendo en la GPU. Los modelos cerrados (Gemini,
Claude, GPT con un millón de contexto) no dicen qué usan.

**Afuera del modelo (el "harness", el sistema que lo rodea).** El modelo tiene su
ventana y listo. Cuando se llena, el sistema resume y descarta (compactación);
guarda cosas en archivos o en una base de búsqueda; y el modelo las trae con
herramientas. Para tareas grandes, lanza sub-agentes con contexto limpio que
vuelven con un resumen. Es un parche a nivel sistema, no parte de la
arquitectura, y es lo que hace cualquier agente actual.

Fijate que el harness se puede poner arriba de **cualquier** modelo, incluido el
nuestro. Por eso la comparación justa es entre arquitecturas, y queda así:

| Corredor | Qué lleva en la cabeza | Rol |
|---|---|---|
| Transformer completo | Todo el libro sobre la mesa | Techo de calidad. No escala. Referencia, no rival |
| Ventana + consultar | Las últimas N palabras tal cual | El parche de hoy. **El rival a vencer** |
| RNN + consultar | Una hoja de apuntes *aprendida* | Nuestra idea |

Las reglas de la carrera: mismo libro guardado afuera, mismo mecanismo para
consultarlo, mismo **presupuesto de consultas** (por ejemplo, podés consultar en
el 5% de los tokens). Si no igualamos eso, no sabemos si ganó la cabeza o el
mecanismo de consulta. Y la pregunta central del proyecto es:

> Con el mismo libro afuera y el mismo presupuesto de consultas, ¿sirve de algo
> tener una cabeza que **resume** en lugar de una que **recorta**?

La hipótesis es que la RNN necesita consultar menos para llegar a la misma
precisión, porque parte de lo que le preguntan ya lo tiene en la hoja. Si las
dos curvas se pegan, el resumen aprendido no aporta nada y la idea muere ahí.

## 5. Cómo se evalúa algo así

### La regla de oro

**Todo fijo, cambiás una sola pieza.** Mismos datos, mismo tamaño de modelo,
mismo tiempo de entrenamiento. Lo único que cambia es la pieza que querés
probar. Como a escala chica los resultados son ruidosos, cada variante se corre
con varios learning rates (la velocidad con la que el modelo ajusta sus
parámetros) y varias semillas, y se reporta el mejor.

### La escalera

Nadie prueba una arquitectura nueva entrenando un modelo grande de entrada. Se
sube por una escalera de tres niveles, y nadie salta al siguiente sin resultados
en el anterior.

| Nivel | Qué evalúa | Con qué datos | Escala | Tiempo por corrida |
|---|---|---|---|---|
| 1. Test de juguete | Solo la habilidad que el cambio promete mejorar | Secuencias generadas por un script | Modelo de 2 capas | Minutos en 1 GPU |
| 2. Modelo chico real | Si el cambio sirve para lenguaje de verdad, sin romper lo demás | Texto de internet filtrado (FineWeb-Edu), 10 a 100 mil millones de tokens | 100M a 1B parámetros | Horas o días en 8 GPUs |
| 3. Contexto largo | Lo que el cambio dice resolver | Textos largos con datos escondidos | El modelo del nivel 2 | Solo evaluar |

En el nivel 2 el puntaje es doble: qué tan bien predice la palabra siguiente (la
"loss") y cuántas respuestas acierta en quizzes estándar. Si el cambio empeora el
lenguaje general, se descarta aunque mejore la memoria.

### Qué hacen los labs grandes

Lo verificamos en los reportes de Kimi Linear, MiniMax-01 y Qwen (detalle en
`notas/como-evaluan.md`). Hay un núcleo común, cambien lo que cambien:

- **La escalera de scaling.** Entrenan la arquitectura nueva y la vieja a 5 o 6
  tamaños distintos (MiniMax: de 70M a 7B), con los mismos datos, y grafican cómo
  baja la loss al gastar más cómputo. La nueva tiene que quedar por debajo en
  todos los tamaños. Es la prueba de "no rompiste el lenguaje en general".
- **Lo específico del objetivo va aparte.** Para memoria: tests sintéticos de 2
  capas (Kimi usó MQAR y parientes) y "aguja en el pajar" a contextos largos.
  MiniMax encontró que su atención lineal pura igualaba a la normal en *todo*
  salvo en aguja en el pajar; ese único quiz los llevó al híbrido.
- **Lo caro lo hacen solo ellos.** Kimi probó las proporciones 1:1, 3:1, 7:1 y
  15:1 a escala real (48B parámetros, 1.4 billones de tokens). Inalcanzable para
  nosotros, y no hace falta para la pregunta que tenemos.

### ¿Vale algo lo que encontremos en miniatura?

A medias, y es el riesgo más grande del proyecto. Lo confiable: **si falla en el
juguete, está muerto.** Lo no confiable: si funciona en el juguete, es
"promete", no "sirve". Hay evidencia de que transfiere cuando el test está bien
diseñado (el test de juguete que usamos predice la brecha en texto real; a
MiniMax un test chico le definió un modelo de 456B). Pero también hay historia
de tests sintéticos que mentían. Por eso la escalera: un hallazgo es interesante
si se sostiene al subir de tamaño, y lo honesto es decirlo en cada resultado.

## 6. MQAR: el juego con el que medimos

**MQAR** son las siglas de *Multi-Query Associative Recall*. *Associative
recall* es "acordarse qué va con qué": te muestro pares, `A 4`, `B 7`, `C 1`, y
después te pregunto `B` y tenés que decir `7`. *Multi-query* es que en la misma
secuencia hay **varias** preguntas, sobre pares distintos, en cualquier orden.
Con una sola pregunta al final, un modelo podría zafar guardando una sola cosa;
con muchas, tiene que haber retenido todos los pares.

```
A 4  B 7  C 1  D 9  E 3   →   C ?  A ?  E ?  B ?
```

El modelo se entrena a predecir el token siguiente, como siempre. El puntaje se
mide solo en los `?`: qué porcentaje de esas respuestas acierta. Las perillas de
dificultad son **cuántos pares hay** (de 4 a 256) y **qué tan grande es la hoja**
(el tamaño del estado). Lo inventaron en el paper Zoology (2023) porque los tests
anteriores tenían una sola pregunta y las arquitecturas los pasaban sin saber
recordar de verdad.

Para nosotros tiene una ventaja extra: sabemos exactamente **en qué momento**
hace falta volver atrás, cuando aparece una pregunta. Así podemos mirar si el
modelo aprendió a consultar ahí y no en cualquier lado.

### El banco: cinco juegos, no uno

MQAR mide lo que la RNN hace *peor*. Para una búsqueda de arquitecturas hace
falta medir también lo que puede hacer *mejor*. Por eso el banco v1 usa cinco
juegos que ya trae Zoology: recuerdo exacto (MQAR), recuerdo con
**actualización** (una clave cambia de valor y vale el último, favorece a las RNN
con olvido), recuerdo con **claves compuestas** (el valor depende de dos tokens),
**paridad acumulada** y **mayoría acumulada** (seguimiento de estado y conteo,
donde a los Transformers les cuesta). En todos, el examen incluye secuencias más
largas que las de entrenamiento. La tarjeta de puntajes tiene una fila por
arquitectura y, por juego, la precisión en cada dificultad, más memoria del
estado, parámetros y tiempo. Hay otro banco parecido, MAD (ICML 2024), con tareas
de copia selectiva y compresión que podríamos sumar en la v2.

### El primer gráfico que buscamos

En el juego de recuerdo exacto: precisión contra cantidad de pares, tres curvas:

- **Atención**: ~100% siempre. Es el techo.
- **Ventana** (últimos N tokens): se cae apenas los pares no entran en la ventana.
- **Gated DeltaNet**: se cae cuando los pares superan lo que le entra en la hoja,
  y más tarde cuanto más grande la hoja. Probamos tres tamaños.

Esto es "reproducir la falla conocida": el punto de partida, no la meta. Sin esa
curva no podemos ver después si consultar la arregla.

### El gráfico objetivo del proyecto

Precisión contra **presupuesto de consultas** (qué porcentaje de los tokens puede
volver a mirar), una curva por cabeza: ventana + consultar vs. RNN + consultar.
Más un control importante: la misma RNN consultando **al azar** con el mismo
presupuesto, para separar "aprendió *cuándo* mirar" de "cualquier mirada sirve".

## 7. Dónde corre y qué cuesta

- **La Mac** (M1 Pro, 16 GB, sin GPU NVIDIA) es solo para escribir código.
  Alcanza para una prueba de humo con atención en CPU.
- **Los experimentos corren en una VM de Azure** que ya existía en la
  suscripción, con 2 GPUs H100. Es "spot": Azure puede
  apagarla si necesita la capacidad, a cambio de cobrar 5 veces menos, unos
  $2.58 por hora. **Apagada no cobra nada.** Una tarde del nivel 1 sale menos de
  $10. Para el nivel 2 habrá que pensar en más GPUs.
- Todo se maneja desde la Mac con `scripts/vm.sh` (`start`, `stop`, `sync`,
  `run`, `fetch`). Regla: apagar al terminar.

## 8. Con qué herramientas

No inventamos ni el dataset, ni la tarea, ni las evaluaciones. Existen:

- **Zoology** (`HazyResearch/zoology`): el generador de MQAR, el entrenamiento,
  los barridos y los modelos para comparar (atención, ventana, Gated DeltaNet y
  otros). Usamos un **fork** nuestro como submódulo, con un cambio chico: guardar
  los resultados en archivos (el original solo los manda a wandb, un servicio
  web). Lo nuestro, el mecanismo de consulta, va en nuestro propio código y
  Zoology lo carga por nombre, sin tocar el fork.
- **flash-linear-attention**: los kernels de Gated DeltaNet para GPU NVIDIA.
- Para el nivel 2, más adelante: `flame` para entrenar en FineWeb-Edu,
  `lm-evaluation-harness` y el repo de Based para los quizzes de memoria exacta,
  RULER para contexto largo.

## 9. Dónde estamos y qué sigue

**Hecho.** El repo, el fork de Zoology, la VM y el tablero. El banco v1 corrió
completo (120 corridas, 0 fallas, 2 horas, ~$5) y quedó **calibrado**: muestra
las diferencias conocidas. Atención perfecta en recuerdo exacto; Gated DeltaNet
se cae donde se le llena la hoja, y más tarde cuanto más grande la hoja; las
ventanas se caen cuando los pares no entran. Y dos hallazgos propios: en el
juego de actualización la RNN le gana a la atención (que sin posiciones
explícitas no sabe cuál valor fue el último), y en paridad la variante de
Gated DeltaNet con autovalores negativos pasa de azar a 100%, generalizando a
8 veces el largo de entrenamiento, mientras atención y la variante normal se
quedan en el azar. Detalle en la bitácora.

**Ahora.** Corren los dos juegos de abstracción (reglas en contexto y mini-ARC
con objetos, 48 corridas), con el examen de instancias reales de 1D-ARC.

**Después.** Diseñar el mecanismo de consulta y meterlo como una variante más en
el mismo barrido. Las preguntas de diseño abiertas, en orden de importancia:

1. **¿Qué dispara la consulta?** Un router aprendido, o una señal gratis: Gated
   DeltaNet ya calcula un error al escribir ("mi memoria no sabía esto").
   Cuidado con la bitter lesson: la regla no la escribimos nosotros, la tiene
   que aprender el modelo.
2. **¿A qué se mira?** Tokens crudos, bloques, resúmenes de bloques, o fotos del
   estado.
3. **¿Qué se hace con lo que trae?** Sumarlo a la salida, o reescribirlo en el
   estado para no tener que buscarlo de nuevo.
4. **¿Cómo se entrena algo discreto?** Lo discreto (consultar o no) no tiene
   gradiente. Hay relajaciones conocidas, o entrenar suave y endurecer después.

**Mapa de documentos.** Este archivo explica. `bitacora.md` registra lo que pasa,
con fecha. `notas/` guarda la investigación cruda: `brief.md` (la conversación
original con todo el trabajo relacionado) y `como-evaluan.md` (cómo evalúan los
papers y los labs, con números y fuentes).
