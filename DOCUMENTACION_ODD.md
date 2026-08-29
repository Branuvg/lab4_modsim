# Documentación ODD del modelo ABM de bicicletas compartidas

## Overview

### Propósito

El modelo responde a la pregunta de cómo cambia la congestión vehicular urbana durante la hora pico cuando se activa una política de bicicletas compartidas que permite que ciertos viajeros sustituyan el automóvil por la bicicleta.

### Entidades, variables de estado y escalas

La implementación operativa contiene dos tipos de agentes:

- `CommutterAgent` representa a una persona que se desplaza hacia un destino del distrito central de negocios. Sus atributos son `ingreso` (Q/mes), `distancia` (km, distancia socioeconómica muestreada al trabajo), `destino` (coordenada de la cuadrícula), `t_salida` (paso entero en el que puede comenzar, entre 0 y 4), `modo` (`"bicicleta"` o `"automovil"`), `vehiculo` (referencia al vehículo asociado, o `None`), `estado` (`"En espera"`, `"Viajando"` o `"Llegado"`), `pasos_espera` (conteo de pasos en que no pudo avanzar) y `t_llegada` (paso de llegada, inicialmente `None`).
- `VehicleAgent` representa el automóvil de un commuter que eligió automóvil. Mantiene `conductor`, la referencia al `CommutterAgent` correspondiente. En la versión operativa no define un `step()` propio ni se añade al scheduler; se desplaza cuando su conductor se desplaza.

El modelo `UrbanMobilityModel` utiliza una `MultiGrid` de `20 x 20` celdas, sin toro (`torus=False`), con una escala declarada de 1 km por celda. El entorno mantiene `congestion_map`, una matriz entera de `20 x 20` que cuenta vehículos por celda; `t`, el paso actual; las listas de `llegados` y `vehiculos`; y registros de activación, historial y cambios de modo. `CAP_CELDA = 3` es el umbral para bloquear el movimiento cuando una celda contiene tres commuters, y también el umbral para considerar saturada una celda del mapa cuando contiene tres vehículos. Un paso no se interpreta como una duración física adicional en horas: el notebook lo usa como unidad discreta de simulación y fija `T_HORIZON = 4` para la medición de hora pico.

La población predeterminada es de 150 commuters. Los atributos `ingreso` y `distancia` se generan conjuntamente con una normal bivariada de media

$$
\boldsymbol{\mu}=(8500,6.5),
\qquad
\boldsymbol{\Sigma}=\begin{pmatrix}4\,000\,000&-800\\-800&4\end{pmatrix},
$$

en unidades de (Q/mes, km). La correlación teórica es `-0.20`. Se remuestrea el vector completo cuando el ingreso no es positivo o la distancia es menor que `DIST_MIN = 0.3` km.

El modelo conserva además un mapa de congestión como variable ambiental: se reinicia y actualiza contando los `VehicleAgent` existentes, no contando directamente bicicletas. La medida `indice_congestion()` es la fracción de las 400 celdas cuyo conteo de vehículos es al menos 3.

### Descripción general del proceso y scheduling

En cada paso de la versión operativa ocurre lo siguiente:

1. Se registra el conjunto de commuters activos al inicio y se reinicia el registro del orden de activación.
2. El scheduler baraja y ejecuta `step()` sobre los agentes registrados en el modelo. En Mesa 2 se usa `RandomActivation`; en el entorno observado (Mesa 3.5.1), la ruta es `self.agents.shuffle_do("step")`. En Mesa 3 los `VehicleAgent` creados durante la corrida también quedan registrados en el modelo y reciben el `step()` heredado de `mesa.Agent`, que no realiza ninguna acción; no se agregan explícitamente al scheduler. La activación de los commuters es asíncrona: cada uno observa el estado disponible en el momento en que es activado y puede modificar su posición antes de otras activaciones.
3. Un commuter aún no sale si `t < t_salida`. En su primera activación posterior elige modo. Si elige automóvil, se crea y coloca en su celda un `VehicleAgent`; si elige bicicleta, no se crea vehículo.
4. El commuter intenta avanzar una celda hacia el destino por un camino Manhattan mínimo. Si todos los candidatos están bloqueados, permanece en su celda e incrementa `pasos_espera`.
5. Si llega a `destino`, se marca como `Llegado`, se registra `t_llegada` y, cuando `eliminar=True`, se retiran él y su vehículo de la cuadrícula, del registro de vehículos y del scheduler.
6. Al terminar todas las activaciones se recalcula `congestion_map`, se añade una observación a `historial` y se incrementa `t`.

El scheduler no es síncrono en la corrida efectiva. El notebook define una variante síncrona de contraste (`_CommutterSincrono`) con `step/advance`, pero no la emplea en los escenarios reportados. Aunque el código etiqueta el scheduling asíncrono aleatorio como “justificado”, no incluye una justificación textual adicional ni ejecuta una comparación de resultados entre ambos schedulers; por ello, la documentación no atribuye al modelo una razón que no esté explicitada en el notebook.

## Design concepts

### Emergencia

La congestión emerge de la acumulación espacial de commuters que avanzan hacia destinos cercanos al centro y de sus vehículos asociados, junto con la capacidad de cada celda. Los movimientos individuales, los tiempos de salida, los empates de ruta y los bloqueos producen concentraciones locales y esperas. La política puede reducir el número de automóviles, pero también modifica qué agentes y vehículos ocupan las trayectorias. En la corrida con semilla 42, el pico de `cong_media` fue 0.3375 sin política y 0.3250 con política; estos son resultados de simulación, no parámetros del modelo.

### Heterogeneidad

La heterogeneidad principal está en ingreso, distancia, destino, origen y tiempo de salida. Ingreso y distancia se generan conjuntamente mediante `Z ~ N(0,I)` y la transformación de Cholesky `X = mu + Z @ L.T`, con `Sigma = L @ L.T`; la covarianza negativa `-800` expresa que, en la distribución especificada, mayores ingresos tienden estadísticamente a asociarse con menores distancias. El notebook calcula la correlación teórica `-0.20` y, para la muestra de semilla 42, obtiene `r = -0.1388` con IC de Fisher `[-0.2925, 0.0220]`.

La variante estocástica de elección de modo también aparece en el notebook: genera una propensión con `Beta(5,2)` y usa `beta0=2.2`, `beta1=-0.55`. Esa variante no es llamada por el `step()` operativo, que usa la regla determinista más la percepción de congestión.

### Estocasticidad

Las fuentes implementadas son:

- el muestreo normal bivariado de ingreso y distancia, incluido el remuestreo por truncación;
- la generación normal del destino alrededor del centro `(width//2, height//2)` con `SD_CBD=2.0`, su redondeo y recorte a la cuadrícula;
- la selección aleatoria del origen entre las celdas a la distancia Manhattan requerida;
- el tiempo de salida uniforme en `{0,1,2,3,4}`;
- el barajado aleatorio del scheduler en cada paso;
- el barajado aleatorio entre los dos candidatos de un paso Manhattan cuando ambos existen.

La semilla se aplica separadamente al generador de Mesa y al generador de NumPy. La variante estocástica de transporte usa además `random()` para comparar con una probabilidad logística, pero no participa en las corridas principales.

### Observación

Durante la simulación se registra en `historial`, después de cada paso, el paso `t`, commuters activos, llegados, vehículos presentes, media y máximo de `congestion_map` y número de celdas con al menos `CAP_CELDA` vehículos. También se conservan el reparto modal, los pasos de espera, los tiempos de viaje, el número de cambios por congestión y los atributos iniciales.

El resultado `Y` de cada corrida es `m.congestion_map.mean()` después de cuatro pasos. Se ejecutan 100 corridas con política y 100 sin política, usando las mismas semillas `42,...,141` en ambos escenarios. Se comparan medias, desviaciones estándar, coeficientes de variación, el tamaño estimado `M*`, histogramas y mapas de congestión. Para la diferencia se define `Delta_Y = Y_sin - Y_con` y se aplica bootstrap pareado sobre los índices de corrida.

Los resultados almacenados en el notebook son: `mu_hat=0.2845` sin política y `0.2768` con política; `Delta_Y=0.0077`, reducción relativa de 2.7 %, error estándar bootstrap 0.0005 e IC percentil 95 % `[0.0067, 0.0087]`. Las gráficas incluyen atributos ingreso-distancia, mapas de congestión en el pico, vaciado de la red, congestión media temporal, histogramas de `Y`, convergencia del error bootstrap e histograma bootstrap de `Delta_Y`.

## Details

### Inicialización

Cada corrida crea `UrbanMobilityModel` con `politica_activa`, `n_commuters=150`, dimensiones `20 x 20`, semilla y, por defecto, `eliminar=True`, `percepcion_congestion=True`, `umbral_congestion=0.8` y `dist_max_bici=6.0`. Se inicializan `congestion_map` y los registros, se generan los 150 pares de atributos por Cholesky y se calcula el destino de cada commuter.

El destino se genera alrededor del centro de la cuadrícula. La distancia discreta usada para construir el origen es

$$
d=\max\left(1,\operatorname{round}(\text{distancia}/1\text{ km})\right).
$$

El origen se selecciona de las celdas que cumplen `|x-x_dest|+|y-y_dest|=d`. Si no existe ninguna dentro de los límites, `d` se reduce hasta encontrar candidatos y se incrementa `n_clipped`. En cada origen se coloca un commuter; los vehículos no se crean en la inicialización, sino en la primera activación de los commuters que eligen automóvil. Inicialmente el mapa de congestión es cero y se actualiza una vez al concluir la inicialización.

Con `seed=42`, la generación de atributos no requirió remuestreo por truncación. La implementación observada corresponde a Mesa 3.5.1 y activa aleatoriamente a los commuters; la verificación del notebook muestra 150 activaciones por paso y exactamente una activación por agente activo.

### Submodelos

#### Elección del modo de transporte

En la primera activación después de la salida, el commuter elige bicicleta si

$$
\text{distancia}\le 3\text{ km}\quad\land\quad\text{politica\_activa}.
$$

En caso contrario, con política activa, puede cambiar a bicicleta si

$$
\text{distancia}\le 6\text{ km}\quad\land\quad
\overline{C}_{\mathcal N}\ge 0.8,
$$

donde `\overline{C}_{\mathcal N}` es la media del `congestion_map` en la vecindad local de Moore de 3x3, recortada en los bordes. Si ninguna condición se cumple, el modo es automóvil. La percepción es local: el agente no consulta el mapa global para decidir.

Con política inactiva todos los commuters eligen automóvil. Para la población fija de la primera demostración, 3 de 150 fueron elegibles por distancia; en la corrida completa con semilla 42, 9 eligieron bicicleta: 3 por distancia y 6 por congestión percibida.

#### Movimiento hacia el destino con distancia Manhattan

`_candidatos_paso()` construye como máximo dos movimientos unitarios: uno que reduce la diferencia absoluta en `x` y otro que reduce la diferencia absoluta en `y`. Por ello cada movimiento reduce la distancia Manhattan al destino en una unidad. La lista de candidatos se baraja y se prueban en ese orden.

#### Manejo de celdas ocupadas y espera

Una celda candidata está bloqueada si contiene al menos `CAP_CELDA=3` objetos `CommutterAgent`; el conteo no incluye directamente los vehículos. El commuter toma el primer candidato no bloqueado. Si ambos están bloqueados, no se mueve y suma uno a `pasos_espera`. Cuando se mueve, su vehículo, si existe, se mueve a la misma celda. Las bicicletas también cuentan como `CommutterAgent` para el bloqueo.

#### Actualización del mapa de congestión

Al final de cada paso se ejecuta `_actualizar_congestion()`: se pone toda la matriz en cero y se incrementa la celda de cada `VehicleAgent` de `self.vehiculos` cuya posición no sea `None`. La percepción usada durante un paso, por tanto, corresponde al mapa actualizado al final del paso anterior (salvo el mapa inicial cero). El scheduler no mueve vehículos por una regla propia: la versión operativa de `VehicleAgent` no tiene `step()` y el vehículo sigue al conductor.

#### Eliminación de agentes al llegar al destino

Cuando `self.pos == self.destino`, `_llegar()` marca el estado como `Llegado`, registra el paso y añade el commuter a `llegados`. Con `eliminar=True`, retira el vehículo de la cuadrícula y de `vehiculos`, retira el commuter de la cuadrícula y lo desregistra del scheduler. La ejecución se detiene cuando no quedan commuters activos o al alcanzar el máximo de 120 pasos.

La guarda inicial de `step()` retorna inmediatamente para un agente ya llegado. Esta guarda evita reactivaciones indebidas si un agente permanece en el scheduler; con la eliminación activa normalmente cada agente se elimina antes de poder ser reactivado en el mismo mecanismo.

#### Cálculo del índice de congestión final

El índice usado en la versión operativa es

$$
I_C=\frac{1}{400}\sum_{c\in C}\mathbf 1\{C_c\ge 3\},
$$

donde `C_c` es el número de vehículos en la celda `c`. Para la medición de escenarios no se usa directamente `indice_congestion()`, sino la media de todos los conteos del mapa después de `T_HORIZON=4` pasos:

$$
Y=\frac{1}{400}\sum_{c\in C} C_c.
$$

#### Ejecución de escenarios, bootstrap e intervalos de confianza

Se ejecutan 100 semillas pareadas por escenario. Para cada escenario se calculan

$$
\hat\mu_Y=\operatorname{mean}(Y_i),\quad
\hat\sigma_Y=\operatorname{sd}(Y_i),\quad
CV=\hat\sigma_Y/\hat\mu_Y,
$$

y

$$
M^*=\left(\frac{1.96\,CV}{0.05}\right)^2.
$$

Los valores obtenidos fueron `M*=3.2` sin política y `M*=3.9` con política.

Para cada vector de 100 resultados, `bootstrap_media()` genera `B=2000` remuestras de tamaño 100 con reemplazo, calcula la media de cada una y obtiene su media, desviación estándar e IC percentil `[2.5,97.5]`. Los intervalos fueron `[0.2820,0.2872]` sin política y `[0.2741,0.2795]` con política.

La curva de convergencia repite el bootstrap con los primeros `M'=10,20,...,100` resultados y `B=5000`; el cambio entre 90 y 100 corridas fue 7.0 % en ambos escenarios. Finalmente, `bootstrap_diferencia_pareada()` remuestrea los mismos índices para `Y_sin` y `Y_con`, calcula en cada remuestra la diferencia de medias y obtiene el IC percentil de `Delta_Y`. El IC no contiene cero.

## Observaciones de consistencia

El notebook contiene una primera definición de las clases con cuadrícula toroidal, movimiento aleatorio de Moore, vehículos creados desde el inicio y un índice basado en `ocupacion`; esas clases son redefinidas posteriormente. Los resultados de movilidad dirigida, mapas, eliminación y escenarios corresponden a la segunda definición, que usa `TORUS=False`, destinos, caminos Manhattan, `congestion_map`, vehículos creados al elegir automóvil y umbral `>= CAP_CELDA`.

También aparece una variante estocástica de elección de modo y una alternativa síncrona, pero no son usadas por las corridas principales. El texto del notebook describe en algunos pasajes la congestión como “fracción de vehículos por celda” y en otros como celdas saturadas; el código distingue ambas medidas: `Y` es la media de vehículos por celda, mientras que `indice_congestion()` es la fracción de celdas con al menos tres vehículos. Finalmente, una nota textual atribuye el ruido principalmente al scheduler y a los empates de camino; el código también incorpora aleatoriedad en la generación de atributos, destinos, orígenes y tiempos de salida.
