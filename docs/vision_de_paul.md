# Miradas Abiertas: Cuantificación del Comportamiento Visual Exploratorio en Fútbol con Datos Posicionales Mejorados por Estimación de Pose

**Joris Bekkers**

Federación de Fútbol de EE.UU., Atlanta, EE.UU.          UnravelSports, Breda, Países Bajos

## 1. Resumen

Los enfoques tradicionales para medir el comportamiento visual exploratorio en fútbol se basan en contar acciones exploratorias visuales (VEAs) basadas en movimientos rápidos de cabeza que exceden 125°/s, pero este método adolece de sesgo por posición del jugador (es decir, un enfoque en centrocampistas centrales), desafíos de anotación, restricciones de medición binaria (es decir, un jugador está escaneando o no), carece de poder para predecir el éxito futuro relevante a corto plazo dentro del juego, y es incompatible con modelos fundamentales de analítica futbolística como el control del campo. Esta investigación introduce una novedosa capa de visión estocástica continua formulaica para cuantificar la percepción visual de los jugadores a partir de seguimiento espaciotemporal mejorado con estimación de pose. Nuestros modelos probabilísticos de campo de visión y oclusión incorporan ángulos de rotación de cabeza y hombros para crear mapas de visión dependientes de la velocidad para jugadores individuales en un plano bidimensional cenital.

Combinamos estos mapas de visión con superficies de control del campo y valor del campo para analizar la fase de espera (cuando un jugador está esperando que el balón llegue después de un pase de un compañero) y su posterior fase con balón. Demostramos que las métricas visuales agregadas - como el porcentaje de área defendida observada mientras se espera un pase - son predictivas del valor de campo controlado ganado al final de acciones de regate utilizando 32 partidos de datos de seguimiento sincronizados mejorados con estimación de pose y datos de eventos con balón de la Copa América 2024. Esta metodología funciona independientemente de la posición del jugador, elimina los requisitos de anotación manual y proporciona mediciones continuas que se integran perfectamente en los marcos analíticos existentes del fútbol. Para apoyar aún más la integración con los marcos analíticos existentes del fútbol, publicamos como código abierto las herramientas necesarias para realizar estos cálculos.

## 2. Introducción

La analítica del fútbol ha evolucionado rápidamente durante la última década, con avances ocurriendo en múltiples dominios diferentes. La investigación de datos a nivel de eventos ha permitido la valoración de jugadores y sus acciones asignando probabilidades a eventos con balón basándose en la probabilidad de que conduzcan a un gol [15], la distinción de roles específicos de jugadores de fútbol [1], la identificación de estilos de juego a partir de patrones de pase utilizando teoría de grafos [8], y la cuantificación de la creatividad en el pase [35]. Simultáneamente, datos de seguimiento sofisticados (como los descritos por [3]) - que capturan el movimiento continuo de jugadores y balón múltiples veces por segundo - han permitido la evaluación riesgo-recompensa de pases [32], marcos de aprendizaje por refuerzo para la toma de decisiones óptima [33], y la evaluación de jugadores sin balón comparando sus movimientos con trayectorias predichas [41]. Estos datos han permitido además: el análisis de formaciones a través de distintas fases de juego [4], al tiempo que posibilitan analíticas defensivas avanzadas incluyendo marcos para cuantificar la intensidad de la presión [5], evaluar las contribuciones individuales de jugadores dentro de situaciones de presión [25], y modelar el éxito de los contraataques [9]. El trabajo fundacional en el área del análisis espacial [40] finalmente evolucionó hacia modelos sofisticados de control del campo [37, 20] - que cuantifican la probabilidad de que cada equipo controle diferentes áreas del campo en cualquier momento dado, y un marco complementario de valor del campo que asigna valor a ubicaciones estratégicamente importantes [20]. El crecimiento explosivo de la investigación, la adopción de datos y la disponibilidad de datos en fútbol ha llevado incluso a la creación de un formato estandarizado para datos de fútbol [2].

Los avances tecnológicos en visión por computador y estimación de pose humana por parte de tecnologías como OpenPose [10], HRNet [39] y HigherHRNet [14] han inaugurado la próxima frontera en la analítica del fútbol. Estos avances han permitido la extracción de información detallada de pose corporal, incluyendo el posicionamiento de cabeza y hombros, permitiéndonos dirigir nuestro enfoque hacia la incorporación de la percepción visual en nuestro modelado futbolístico.

Los enfoques tradicionales para comprender la percepción visual en el fútbol se han apoyado fuertemente en entornos de laboratorio controlados utilizando tecnología de seguimiento ocular para examinar comportamientos de la mirada y patrones de búsqueda visual [11, 36, 41, 30]. Aunque estos estudios han proporcionado valiosas ideas sobre los procesos perceptivo-cognitivos subyacentes al rendimiento experto - demostrando que los jugadores habilidosos emplean estrategias de búsqueda visual más eficientes con duraciones de fijación más cortas a través de ubicaciones más relevantes [36, 42] - están fundamentalmente limitados por sus condiciones experimentales. Estos estudios de laboratorio típicamente requieren que los participantes vean grabaciones de vídeo y tomen decisiones sobre situaciones de juego hipotéticas en lugar de medir el comportamiento visual durante el juego real [29]. Una revisión sistemática reveló que ningún estudio ha investigado los comportamientos de exploración durante situaciones reales de juego abierto, con la mayoría de la investigación sin reflejar las demandas complejas y dinámicas del rendimiento real en fútbol [29]. Mientras que otra revisión, cubriendo desde 2016 hasta 2022, muestra un progreso alentador hacia condiciones de investigación más realistas y un mayor énfasis en respuestas motoras naturales. Sin embargo, estos avances han sido acompañados por limitaciones metodológicas persistentes, incluyendo tamaños de muestra más pequeños, tecnología de seguimiento ocular obsoleta y estándares insuficientes de reporte de calidad de datos [24].

La traducción de estos hallazgos de laboratorio a la medición en campo se ha basado principalmente en contar *acciones exploratorias visuales* (VEAs) basadas en movimientos rápidos de cabeza que exceden 125°/s [22]. Sin embargo, este enfoque presenta varias limitaciones fundamentales que restringen su aplicabilidad al análisis de rendimiento. Las VEAs demuestran un sesgo inherente hacia posiciones de juego que requieren acciones exploratorias visuales frecuentes (p. ej., centrocampistas centrales) [16, 22, 23, 28], son difíciles de anotar con precisión con una variabilidad inter-observador sustancial [26, 13, 23], resultan difíciles de recopilar de manera fiable a partir de datos de seguimiento estándar de 25 FPS [26], y se basan en mediciones binarias (es decir, movimiento rápido de cabeza, o no) que no logran acomodar inexactitudes en los datos ni capturar la naturaleza continua de la atención visual [26, 13]. Además, los estudios que validan las VEAs como métrica de rendimiento se basan en medidas de juego simplistas como el éxito de pase o el éxito de pase hacia adelante [23, 28]. Estas asociaciones ni siquiera son estadísticamente significativas para algunos grupos de posiciones, socavando la utilidad de la métrica [23].

Adicionalmente, la frecuencia de VEA varía significativamente según factores contextuales como la posición en el campo, la fase de juego y la presión del oponente [27, 31]. Además, un estudio reciente [12] no encontró diferencias significativas en VEA entre jugadores súper élite y élite, cuestionando la premisa fundamental de que la frecuencia de VEA sirve como un diferenciador de rendimiento fiable.

Esta investigación introduce una novedosa integración de datos de estimación de pose y modelado espacial para abordar las limitaciones de la medición tradicional de VEA. Introducimos una capa de *visión* que utiliza datos de seguimiento posicional mejorados con ángulos de rotación de cabeza y hombros extraídos de estos datos estimados de pose corporal. Utilizamos esta capa de visión estocástica - que representa la distribución de probabilidad de dónde cada jugador está dirigiendo su atención visual a lo largo del campo - para cuantificar el comportamiento de *escaneo* de forma continua para cada jugador individualmente, en lugar de simplemente contar movimientos rápidos de cabeza.

Al combinar nuestra capa de visión con los marcos establecidos de *control del campo* y *valor del campo* [20], demostramos que modelar esta capa de visión estocástica puede ser altamente valioso para cuantificar el comportamiento visual exploratorio. Incluso las métricas visuales agregadas (como el porcentaje de área defensiva observada mientras se espera un pase de un compañero) son fuertes predictores del valor de campo ganado (o perdido) al final del regate posterior del jugador.

Nuestro enfoque aborda las limitaciones fundamentales de la investigación contemporánea sobre VEA estableciendo un método formulaico que funciona independientemente de la posición del jugador, elimina la dependencia de la anotación manual de datos y proporciona mediciones continuas en lugar de binarias. Al definir el comportamiento visual exploratorio como una cuadrícula estocástica bidimensional que abarca el campo de juego, creamos compatibilidad directa con marcos analíticos existentes como *pitch control* y SoccerMap [19], permitiendo una integración perfecta en las metodologías actuales. Adicionalmente, demostramos que los métodos tradicionales de conteo de VEA - capturados a 25 FPS - carecen de poder predictivo para resultados de rendimiento con balón posteriores, destacando la necesidad de enfoques más sofisticados para cuantificar la percepción visual en el fútbol y cuestionando aún más la fiabilidad de las VEAs como indicadores de rendimiento.

### 2.1. Investigación

Comenzamos modelando la capa de visión para un jugador individual (detallado en la Sección 3.1). Esta consiste en dos componentes: un **mapa de campo de visión** probabilístico, y un **mapa de oclusión** probabilístico creado por todos los demás jugadores en el campo tal como se percibe desde la perspectiva de este jugador individual (Sección 3.1). Para fomentar la reproducibilidad y apoyar más investigación sobre comportamiento visual exploratorio, compartimos código y datos para construir estos mapas en el GitHub de la Federación de Fútbol de EE.UU.¹

Combinamos estos mapas - usando multiplicación matricial elemento a elemento - con dos componentes de Fernández & Bornn (2018) [20]: un mapa probabilístico de *control del campo* que aproxima el espacio controlado por ambos equipos, atacante y defensor, y un mapa de *valor del campo* que describe dinámicamente el valor de cada segmento del campo aprendido de las configuraciones del equipo defensor dada la ubicación del balón. Adicionalmente introducimos un parámetro de modificación al cálculo de la superficie de control del campo para reducir la cantidad de espacio controlado por un jugador. Usamos esto como un proxy para las ubicaciones en el campo que un jugador puede alcanzar en un período muy corto de tiempo, lo que llamamos **control de campo inminente**. Combinar estos aspectos nos da la capacidad de medir el comportamiento visual exploratorio durante un partido para cualquier jugador, en cualquier momento del juego, y relacionarlo directamente con el valor de campo ganado o perdido.

Para validar nuestros métodos construimos un conjunto de modelos XGBoost de Clasificación Binaria para aprender si el comportamiento visual exploratorio modelado es predictivo del éxito futuro a corto plazo dentro del juego, medido como un aumento o disminución significativo del valor de campo controlado al inicio y al final de una fase (ver

¹ https://github.com/USSoccerFederation/ssac26_visual_exploratory_behavior

Sección 4). Nos enfocamos específicamente en la fase de *espera* y la posterior fase *con balón*. La fase de espera se define como los momentos en que un jugador está esperando que el balón llegue después de que su compañero ha ejecutado un pase y la posterior fase *con balón* es simplemente el movimiento con balón (p. ej., regate) que el jugador ejecuta después de recibir el balón. La combinación de visión y valor del campo nos da un enfoque integral para modelar la capacidad de los jugadores individuales para identificar visualmente el espacio valioso y actuar en consecuencia en momentos clave de la cadena de posesión de un equipo.

### 2.2. Conjunto de Datos de Validación

Para validar esta investigación, utilizamos 32 partidos de datos de seguimiento de transmisión proporcionados por Respo.Vision [34] a todos los equipos participantes en la Copa América 2024. Estos datos (registrados a 25 fotogramas por segundo) contienen identificadores de equipo y jugador, y coordenadas *x* e *y* de jugadores y balón. Están mejorados con ángulos de rotación de cabeza, hombros y cadera de cada jugador en el plano bidimensional. La Figura 1 muestra una comparación bidimensional cenital de una instantánea de datos de seguimiento regulares comparados con su contraparte mejorada con seguimiento de extremidades.

Para complementar estos datos, y para extraer las fases relevantes (*espera* y *con balón*), los datos posicionales se mejoran con datos de eventos sincronizados de StatsPerform (Opta) [38] usando una combinación de *kloppy* [44], *socceraction* [15] y una versión simplificada del algoritmo de sincronización basado en reglas *ETSY* [43]. Las recepciones de balón se identifican como el primer mínimo local de la distancia entre el jugador que recibe el balón y el balón.

En total, nuestro conjunto de datos consiste en aproximadamente 2,3 millones de fotogramas de datos de seguimiento y más de 60.000 eventos. Cada evento se asigna a un único fotograma de datos de seguimiento. Filtramos estos datos para incluir solo secuencias no disputadas de juego abierto, definidas como una secuencia de juego que tiene al menos 1 pase exitoso y como cualquier acción que ocurre al menos 7 segundos después de la jugada a balón parado más reciente. Esto produce aproximadamente 14.000 momentos de espera con eventos posteriores con balón.

**(a)** Regular                                              **(b)** Mejorado con pose

**Fig.1.** Instantáneas de datos de seguimiento de transmisión de Respo.Vision de Argentina v Canadá en la Copa América 2024 excluyendo (a) e incluyendo (b) la orientación de cabeza y hombros.

Los ángulos faltantes (cabeza, hombros o cadera) se interpolan por jugador usando la fórmula de Euler, que descompone los ángulos en componentes reales e imaginarios que se interpolan independientemente y luego se reconstruyen en ángulos. Este enfoque es necesario porque los ángulos son cíclicos, requiriendo interpolación a lo largo del camino angular más corto.

## 3. Metodología

En esta sección, iluminamos todos los aspectos necesarios para construir un mapa de visión integral para superponer sobre modelos de control del campo y valor del campo. Como suplemento visual a esta sección, cinco vídeos cortos están disponibles en [7] y ². Adicionalmente, una tabla de referencia que define toda la notación matemática utilizada en esta investigación se puede encontrar en el Apéndice A.

### 3.1. Visión

Para construir una representación bidimensional cenital integral de la percepción visual de un jugador, creamos dos modelos complementarios: un modelo de **visión** (ver Sección 3.1.1) para cuantificar la capacidad de un jugador para ver a otros jugadores en el campo, y un modelo de **oclusión** (ver Sección 3.1.2) para describir a cualquier jugador en el campo bloqueando (partes de) la visión de un jugador. Juntas, estas dos facetas nos darán el mapa de visión completo de cada jugador expresado como una cuadrícula de 105x68 (es decir, el largo y ancho del campo).

Debido a que la visión humana no es perfecta, y los objetos en la visión periférica, o aquellos a mayor distancia, son más difíciles de localizar exactamente, especialmente a altas velocidades, introducimos un método probabilístico para cuantificar tanto los modelos de visión como de oclusión. Usamos la orientación de la cabeza como proxy para la orientación de los ojos, ya que frecuentemente están orientados en la misma dirección según lo reportado por [18, 13, 26]. Aproximadamente el 92% de los cambios de mirada están en la misma dirección que el movimiento de cabeza dentro del plano horizontal [18].

#### 3.1.1 Campo de Visión

Para empezar, creamos un mapa del *campo de visión binario* (V_β) para el jugador *i*, como se muestra en la Figura 2a. Asumimos que la visión binocular abarca 120° [21] y asumimos visión finita hacia los bordes del campo.

Posteriormente, introducimos una capa probabilística sobre este campo de visión binario que debería asemejarse más a lo preciso que un jugador puede identificar la ubicación de otros jugadores dada la incertidumbre inherente. Denotamos esto como el *campo de visión* (V_ρ) del jugador *i* en la ubicación *p*, en el tiempo *t* con ángulo de cabeza θ_h y velocidad *v* (ver Fórmula 1).

$$V_ρ(p, t, θ_h, v)_i = R(c_r(v_i)) ⊙ A(c_a(v_i)) ⊙ V_β(p, t, θ_h)_i \quad (1)$$

$$R(c_r(v_i)) = exp\left[-c_r \left(\frac{d}{σ_r}\right)^2\right] \quad (2)$$

$$A(c_a(v_i)) = exp\left[-c_a \left(\frac{θ_a}{σ_a}\right)^2\right] \quad (3)$$

² https://unravelsports.com/ssac/26/veb.html

**(a)** Campo de Visión Binario (V_β)                    **(b)** Campo de Visión (V_ρ) con v_i = 1m/s

**(c)** Campo de Visión (V_ρ) con v_i = 3m/s           **(d)** Campo de Visión (V_ρ) con v_i = 9m/s

**Fig.2.** Un jugador individual *i* en la ubicación (22, 34) con ángulo de cabeza 31°

Introducimos *R* y *A* (Fórmulas 2 y 3) para describir la visión del jugador *i* como una Gaussiana que decrece a medida que la ubicación observada se aleja del jugador *i* radial y angularmente. Tanto *R* como *A* están directamente influenciados por la velocidad del jugador (v_i) - porque asumimos que velocidades más altas resultan en una pérdida de conciencia espacial y un enfoque más estrecho en el entorno inmediato del jugador - a través de los parámetros de escalado c_r y c_a respectivamente (ver Apéndice B). Aquí, c_r determina la tasa de decaimiento de la Gaussiana, *d* denota la distancia desde el jugador *i*, y la constante σ_r es la desviación estándar de la profundidad de visión. *A* a su vez describe una Gaussiana que explica la amplitud de la visión, donde c_a es un parámetro de escalado que controla el decaimiento de la visión a medida que los objetos se mueven más hacia la visión periférica del jugador *i*, θ_a denota el ángulo desde el punto focal y la constante σ_a es la desviación estándar de la visión angular.

Ejemplos de V_ρ con diferentes valores de velocidad *v* se pueden encontrar en las Figuras 2b, 2c y 2d.

#### 3.1.2 Oclusiones

El jugador *i* no es el único jugador ocupando el campo, y como resultado, esos otros jugadores *J* podrían obstruir la visión del jugador *i*. Por lo tanto, mejoramos la veracidad del mapa de campo de visión (V_ρ) del jugador *i* incorporando obstrucciones visuales causadas por otros jugadores en el campo en forma de un mapa de oclusión (V_Φ). V_Φ puede entenderse como la combinación de cada mapa de oclusión individual (V_ϕ,i,j) del jugador *j* bloqueando partes de la visión del jugador *i*. El tamaño de la oclusión causada por el jugador *j* está influenciado por la distancia entre jugadores y la rotación de hombros del jugador *j*. Esta rotación de hombros (θ_s) se usa para determinar cuánta obstrucción forma el jugador *j* dada su rotación hacia el jugador *i*. Si un jugador está orientado de lado desde la perspectiva del jugador *i*, ocupa menos espacio en el campo de visión del jugador *i* comparado con mirar directamente a su torso.

Un mapa de oclusión V_ϕ,i,j depende del ángulo de cabeza (θ_h) y la ubicación del jugador *i* (ahora explícitamente denotada como p_i), y la ubicación (p_j) y el ángulo de hombros (θ_s) del jugador *j*.

Para construir un único mapa de oclusión (V_ϕ,i,j) que describa las áreas obstruidas causadas por el jugador *j* tal como las observa el jugador *i*, creamos un *rayo* probabilístico (Q_i,j) proyectado desde el jugador *i* a través del jugador *j* que sigue la Fórmula 5, y se representa en la Figura 3a. Controlamos la probabilidad máxima de obstrucción con el parámetro α.

Adicionalmente, construimos una máscara binaria V_o,i,j que modela la vista no obstruida entre los jugadores *i* y *j* (ver Figura 3b). Multiplicar estos tres componentes produce el mapa de oclusión V_ϕ,i,j como se describe en la Fórmula 4 y se representa en la Figura 3c. Así, esto modela el espacio que el jugador *i* es (parcialmente) incapaz de observar como resultado del jugador *j*.

**(a)** Rayo (Q_i,j) α                  **(b)** Máscara Binaria (V_o,i,j)               **(c)** Mapa de Oclusión (V_ϕ,i,j)

**Fig.3.** Un mapa de oclusión individual (V_ϕ,i,j) del jugador *j* tal como lo percibe el jugador *i*.

$$V_{ϕ,i,j}(p_i, θ_h, p_j, θ_s, t) = Q_{i,j}(p_i, p_j, θ_s, t) ⊙ V_o(p_i, θ_h, p_j, t) \, α \quad (4)$$

Q_i,j usa el mismo enfoque angular que *A* (en la Fórmula 3) con θ_q y σ_q imitando la funcionalidad de θ_a y σ_a, respectivamente. Sin embargo, ahora el parámetro de escalado c_q (Fórmula 6) está determinado por el ancho angular aparente del jugador *j* desde la perspectiva del jugador *i*, considerando la rotación de hombros del jugador *j* (θ_s). Aquí δ_i,j es la distancia entre los jugadores *i* y *j*, y ω_α es el ancho angular aparente del jugador *j* tal como lo percibe el jugador *i* (derivado del ancho de hombros ω_s y la profundidad del torso d_s. Ver Apéndice C para una derivación completa).

$$Q_{i,j}(p_i, p_j, θ_s, t) = exp\left[-c_q \left(\frac{θ_q}{σ_q}\right)^2\right] \quad (5)$$

$$c_q(p_i, p_j, θ_s, t) = \frac{δ_{i,j}}{ω_α} \quad (6)$$

En esencia, esto significa que ω_α modela cuánta obstrucción forma el jugador *j* dada su rotación hacia el jugador *i*.

El parámetro δ_i,j asegura que el ancho aparente del jugador *j* se escale con la distancia, porque los objetos que aparecen más cerca del jugador *i* deberían proyectar una sombra más grande que aquellos a mayor distancia. Este concepto se ilustra con tres ejemplos en la Figura 4.

$$V_{Φ,i} = \prod_{j \in J} (1 - V_{ϕ,i,j}) \quad (7)$$

Finalmente, modelamos V_Φ como el producto elemento a elemento de todas las matrices de oclusión para el jugador *i* observando a todos los jugadores *J*, como se muestra en la Fórmula 7 y en la Figura 5.

**Fig.4.** Tres ejemplos de mapas de oclusión para demostrar cómo el rayo Q_i,j es apropiadamente controlado por el parámetro de escalado c_q para nunca ser más ancho de lo necesario.

#### 3.1.3 Mapa de Visión Completo del Jugador *i*

Combinamos el mapa de campo de visión (V_ρ,i) y el mapa de oclusión (V_Φ,i) mediante multiplicación matricial elemento a elemento para obtener el mapa de visión total (V) para el jugador *i* siguiendo la Fórmula 8.

$$V_i = V_{Φ,i} ⊙ V_{ρ,i} \quad (8)$$

La Figura 7 muestra dos ejemplos del mapa de visión total a dos velocidades diferentes (v_i = 1m/s y v_i = 9m/s) y la Figura 6 muestra al jugador argentino Rodrigo de Paul observando jugadores directamente después de recibir el balón en un momento durante la final de la Copa América. Cinco vídeos de esta secuencia de juego se comparten en [7] y ².

**Fig.5.** Mapa de oclusión combinado (mostrado como 1 - V_Φ,i) donde el jugador *i* (rojo) observa a 6 jugadores.

**Fig.6.** Rodrigo De Paul observando jugadores en la final de la Copa América [7].

**(a)** Mapa de Visión (V_i) con v_i = 1m/s                **(b)** Mapa de Visión (V_i) con v_i = 9m/s

**Fig.7.** Dos ejemplos del mapa de visión final (V_i) con diferentes velocidades del jugador (v_i).

### 3.2. Control del Campo

Ahora que hemos expuesto el modelo de visión, modelamos la cantidad y calidad del espacio ocupado para cada jugador individual. Logramos esto apoyándonos en dos conceptos de Fernández & Bornn (2018) [20], a saber, *control del campo*; el espacio más probablemente controlado por cualquier equipo (o jugador) en un momento dado, y *valor del campo*; el valor del espacio en el campo.

#### 3.2.1 Control de Campo Inminente

Fernández & Bornn introducen su marco de *control del campo (PC)* para cuantificar la probabilidad de que un jugador controle una ubicación en el campo en cualquier momento dada su ubicación actual, velocidad, magnitud y distancia al balón asignando una distribución bi-normal escalada y rotada para determinar qué jugador podría alcanzar un área primero de manera realista. La Figura 8 muestra su modelo original aplicado a un único fotograma de datos de seguimiento.

Usamos su enfoque e introducimos un parámetro de escalado (c_in) a la fórmula del radio de influencia del jugador (R_i, ver [20]) para disminuir el radio de control de cada jugador (ver Figura 9). Esta operación nos permite imitar el espacio al que un jugador podría moverse en un corto período de tiempo (**control de campo inminente**). Por simplicidad, denotamos esto como PC^(c_in) (p. ej., PC^0.5 para una superficie más pequeña, como se muestra en la Figura 9).

Además, definimos el equipo defensor (Figura 10a), el equipo atacante (Figura 10b) o todos los jugadores excepto el jugador *i* (Figura 10c) para determinar el espacio inminente controlado por cada entidad individualmente.

**Fig.8.** Control del Campo por Defecto                        **Fig.9.** Control del Campo "Inminente" (c_in=0.5)

**(a)** Equipo atacante excl. jugador *i* (PC^0.5_J_att donde i ∉ J)

**(b)** Equipo defensor (PC^0.5_J_def)

**(c)** Jugador atacante *i* (PC^0.5_i)

**Fig.10.** Control de campo inminente para el jugador *i* representado por la flecha azul, compañeros (azul) y oponentes (rojo).

#### 3.2.2 Valor del Campo

Fernández & Bornn adicionalmente introducen una red neuronal de propagación hacia adelante entrenada para estimar la influencia del campo del equipo defensor dada la ubicación del balón. Usan esto para estimar la ubicación de espacio de ataque valioso en el campo, bajo la suposición de que en promedio el equipo defensivo se posiciona en relación al balón para cubrir los espacios más valiosos. Para obtener este *valor del campo* (V_l) entrenamos una red neuronal de propagación hacia adelante similar sobre pares de ubicación de balón y superficies de influencia defensiva asociadas para un subconjunto aleatorio (n=19.504, menos del 1%) de momentos de juego abierto de nuestro conjunto de datos de la Copa América (ver Figura 11a para un ejemplo de predicción).

$$\hat{V}_l = V_l ⊙ V_η \quad (9)$$

En la Figura 11b mostramos la *superficie de normalización de valor del campo por distancia a la portería* (V_η) propuesta por [20]. Esto captura la comprensión intuitiva de que el espacio más cercano a la portería del oponente es más valioso. La Figura 11c muestra el valor del campo normalizado dada la ubicación del balón p_b que se calcula usando multiplicación matricial elemento a elemento siguiendo la Fórmula 9.

**(a)** Superficie de valor del campo (V_l) dada la ubicación del balón p_b (en negro)

**(b)** Superficie de normalización de valor del campo (V_η)

**(c)** Superficie de valor del campo normalizada (V̂_l) dada la ubicación del balón p_b

**Fig.11.** Valores de campo predichos en un rango entre 0 y 1.

## 4. Validación

Ahora evaluamos nuestro modelo de visión realizando un estudio de ablación para evaluar si ver ciertos espacios (p. ej., control de campo inminente atacante observado, o control de campo inminente defensor observado, ver Figura 12) durante la fase de *espera* lleva a los jugadores a alcanzar posteriormente espacio más valioso al final de su fase *con balón*. El conjunto de datos abarca todos los fotogramas relevantes (n=171.318) de nuestros 14.000 momentos de espera alineados con el fotograma final con balón después de esta fase de espera.

**(a)** Control de campo inminente (PC^0.5) del equipo defensor J_def en el tiempo *t*.

**(b)** Mapa de visión (V) del jugador *i* en el tiempo *t*

**(c)** Control defensivo observado V_i(t) ⊙ PC^0.5_J_def(t)

**Fig.12.** Un ejemplo de combinación de múltiples modelos.

Construimos cuatro modelos XGBoost de Clasificación Binaria (ver Sección 4.2) entrenados en conjuntos de datos balanceados con características estandarizadas. Las etiquetas binarias (ver Sección 4.1) describen si el valor del campo del jugador *i* en el tiempo *t* ha aumentado o disminuido significativamente al final de la fase con balón (en el tiempo t^fin_con_balón).

### 4.1. Etiquetas

El aumento en el valor del campo se mide como la razón (p_rat) del valor instantáneo del campo (P_v(t)) del espacio ocupado por el jugador *i* durante la fase de espera en el tiempo actual *t* y el valor instantáneo del campo ocupado por el mismo jugador durante el final de la fase posterior (t^fin_con_balón), como se muestra en la Fórmula 11. En otras palabras, supongamos que un jugador en el tiempo *t* ("ahora") está *esperando* la llegada de un balón, si controlaron significativamente más valor de campo inminente al final de la fase posterior con balón, esta muestra se anota con una etiqueta positiva.

$$p_{rat}(t) = \frac{P_v(t^{fin}_{con\_balón})}{P_v(t) + P_v(t^{fin}_{con\_balón})} \quad (11)$$

Ahora aplicamos estas etiquetas binarias a cada muestra siguiendo la Fórmula 12. Aquí y = 0 constituye una clara disminución en el valor del campo e y = 1 constituye un claro aumento en el valor del campo. Elegimos omitir muestras con un p_rat entre 0,35 y 0,65, porque estos momentos introducen ruido significativo.

$$y = \begin{cases} 0 & \text{si } p_{rat} < 0.35 \\ Excl. & \text{si } 0.35 \leq p_{rat} \leq 0.65 \\ 1 & \text{si } p_{rat} > 0.65 \end{cases} \quad (12)$$

### 4.2. Modelos de Ablación

El estudio de ablación considera cuatro modelos, construidos sobre cuatro conjuntos de datos. Al añadir progresivamente más características a conjuntos de datos subsecuentes podemos cuantificar la contribución específica de nuestras características basadas en visión al rendimiento general del modelo. Cada modelo emplea 150 estimadores, una tasa de aprendizaje de 0,05, profundidad máxima de 4, y parada temprana para prevenir el sobreajuste. Los conjuntos de datos se dividen en conjuntos de prueba y entrenamiento por partido para asegurar que no haya fuga de datos entre conjuntos. A continuación se presenta un esquema de las cuatro configuraciones de modelos.

- **Modelo base** incluye solo la distancia al centro de la portería.
- **Modelo VAE tradicional** añade el número total de VEAs regulares (cuando la velocidad angular de la cabeza del jugador excede 125°/s) en el último segundo [26, 13], y en los últimos dos segundos, y entre t^inicio_espera y el tiempo *t*.
- **Modelo regular** añade además la distancia a la línea de gol; la distancia al centro del campo en las direcciones *x* e *y*; los componentes de los vectores de velocidad (v_x y v_y); y etiquetas de posición estandarizadas derivadas usando la Identificación Elástica de Formación y Posición (EFPI) [6]. Después de asignar las etiquetas de posición las simplificamos en una de seis etiquetas generales de posición, a saber, Centrocampista Abierto, Lateral, Delantero Centro, Centrocampista Central, Central y Extremo.
- **Modelo de visión** añade características agregadas de combinaciones de los modelos descritos en la Sección 3. Distinguimos entre varios componentes clave: superficies de ataque y defensa como se ejemplifica en las Figuras 10a y 10b (p. ej., PC^0.5_J_def); la cantidad de espacio visto usando el *mapa de visión* V_i; la razón de espacio ocupado observado respecto al espacio ocupado total (V_rat, Fórmula 10); la razón de defensa a ataque visto; la cantidad de ataque (o defensa) observada como porcentaje del campo completo; y finalmente, los valores medios de la(s) superficie(s) mencionadas durante el período de tiempo desde t^inicio_espera hasta el tiempo *t*.

$$V^{def}_{rat} = \frac{\sum PC^{0.5}_{J_{def}} ⊙ V_i}{\sum PC^{0.5}_{J_{def}}} \quad (10)$$

### 4.3. Resultados de Validación

La Tabla 1 muestra los resultados de los cuatro modelos progresivamente más complejos de nuestro estudio de ablación. Podemos ver claramente el valor añadido de nuestras características de visión agregadas, ya que mejora el rendimiento del modelo regular de 0,744 a 0,788 AUC.

Además, vemos que añadir características de VAE tradicional al modelo base no produce ningún aumento de rendimiento. Apoyando aún más los hallazgos de [12] que cuestionan las VAEs tradicionales como un diferenciador de rendimiento fiable.

**Tabla 1.** Una visión general de las métricas de rendimiento para los modelos XGBoost individuales

| Modelo | AUC | Precisión | Recall | F1 |
|---|---|---|---|---|
| Base | 0,664 | 0,61 | 0,72 | 0,66 |
| VEA Tradicional | 0,654 | 0,60 | 0,74 | 0,66 |
| Regular | 0,744 | 0,69 | **0,78** | **0,74** |
| **Visión** | **0,788** | **0,71** | **0,78** | **0,74** |

### 4.4. Características con Impacto

Para evaluar cómo las características individuales influyen en el rendimiento del modelo en nuestra configuración de validación, se calcularon valores SHAP para todas las características. Los resultados se muestran en la Figura 13. Azul (positivo) indica que valores más altos para estas características empujan las predicciones hacia la clase positiva (es decir, aumentando el valor de campo inminente del jugador *i* al final de su fase *con balón*). En contraste, rojo indica que valores más altos para estas características empujan las predicciones hacia la clase negativa.

En la parte superior de la Figura 13 notamos que ver una mayor cantidad de espacio ocupado defensivo como porcentaje del área total del campo promediado durante el tiempo de la fase de *espera* [A], ver una mayor proporción de espacio ocupado defensivo comparado con espacio ocupado atacante promediado en el tiempo durante la fase de *espera* [B], ver una mayor cantidad de espacio ocupado atacante como porcentaje del área total del campo promediado en el tiempo durante la fase de *espera* [C] y observar más de las áreas controladas de ataque totales promediadas en el tiempo durante la fase de *espera* [D] son todos predictivos de un aumento en el valor de campo inminente al final de la fase *con balón*.

En la parte inferior de la Figura 13 vemos que valores altos para observar más de las áreas controladas defensivas totales promediadas en el tiempo durante la fase de *espera* [G], y el porcentaje de espacio controlado defensivo observado instantáneamente [H] son ambos predictivos de una disminución en el valor de campo inminente al final de la fase *con balón*.

Las dos últimas características potencialmente nos muestran que observar más espacio controlado por la defensa podría indicar que nuestro jugador atacante está simplemente en un área altamente disputada del campo, en lugar de enseñarnos algo sobre la calidad de su percepción visual. La característica [F] nos dice que simplemente observar más del campo no conduce a mejores resultados. Y lo más importante, las características [A], [B], [C] y [D] indican claramente que los jugadores que hacen observaciones de mayor calidad (es decir, observan mayores cantidades del espacio total ocupado, con un ligero enfoque en identificar espacio controlado defensivamente) mientras esperan un pase, tienen más probabilidades de aumentar la calidad del espacio que ocupan al final de su acción posterior con balón.

**Fig.13.** El impacto y magnitud de las variables individuales en el Clasificador XGBoost que incluye todas las características, usando valores SHAP. En negrita todas las características de visión agregadas creadas usando mapas de visión etiquetados de A a H. Azul (positivo) indica que valores más altos para las variables (p. ej., una mayor distancia a la línea de gol del oponente) empujan la predicción hacia la clase positiva (es decir, un aumento en el valor del campo en el tiempo t^fin_con_balón). Rojo (negativo) indica que valores crecientes para estas variables (p. ej., una mayor distancia desde el centro del campo) empujan la predicción hacia la clase negativa (es decir, una disminución en el valor del campo en el tiempo t^fin_con_balón).

En el medio de la Figura 13 observamos que el número total de VEAs tradicionales - observadas a 25 FPS - hasta el tiempo *t* no tiene impacto en nuestra capacidad de predecir si un jugador va a ganar (o perder) valor de campo al final de la fase *con balón*.

Con la excepción del Extremo, los tipos de posición en general tienen poco o ningún impacto en nuestra capacidad de predecir el cambio en el valor del campo al final de esta fase. Además, el porcentaje de espacio controlado atacante observado instantáneamente (es decir, en el tiempo *t*) [E], y la cantidad promedio del área total del campo observada hasta el tiempo *t* en la fase de *espera* [F], tienen muy poca influencia en el resultado del modelo.

Una explicación para la aparente contradicción en distancia a la portería del oponente y distancia a la línea de gol del oponente se da en el Apéndice C.

## 5. Conclusión

Dentro de esta investigación hemos desarrollado un enfoque novedoso para modelar el comportamiento perceptual visual en el fútbol usando un plano bidimensional cenital con datos de seguimiento posicional mejorados con estimación de pose. Este método integra los marcos existentes de control del campo y valor del campo con un mapa de visión y un mapa de oclusión para cuantificar el valor del campo controlado y observado. Dentro de nuestra validación demostramos que el comportamiento visual durante la fase de espera muestra poder predictivo para cambios posteriores en el espacio controlado después de completar una acción de regate. Específicamente, los jugadores que observan mayores cantidades de espacio ocupado - particularmente áreas controladas defensivamente - mientras esperan pases exhiben mayores ganancias en su posicionamiento espacial después de acciones posteriores con balón, mientras que los métodos tradicionales de conteo de acciones exploratorias visuales (VEA) no tienen ningún valor predictivo para estos resultados. En contraste con las VEAs, nuestro enfoque funciona independientemente de la posición del jugador, elimina los requisitos de anotación manual y proporciona mediciones continuas que se integran con marcos analíticos existentes como control del campo, valor del campo y SoccerMap. A medida que los datos de estimación de pose se vuelven cada vez más disponibles en los deportes de invasión, este enfoque tiene aplicaciones potenciales más allá del fútbol, incluyendo fútbol americano y baloncesto.

## Referencias

[1] Aalbers, B., Van Haaren, J.: Distinguishing between roles of football players in play-by-play match event data. In: Machine Learning and Data Mining for Sports Analytics: 5th International Workshop, MLSA 2018, Co-located with ECML/PKDD 2018, Dublin, Ireland, September 10, 2018, Proceedings 5. pp. 31–41. Springer (2019)

[2] Anzer, G., Arnsmeyer, K., Bauer, P., Bekkers, J., Brefeld, U., Davis, J., Evans, N., Kempe, M., Robertson, S.J., Smith, J.W., et al.: Common data format (cdf): A standardized format for match-data in football (soccer). arXiv preprint arXiv:2505.15820 (2025)

[3] Bassek, M., Rein, R., Weber, H., Memmert, D.: An integrated dataset of spatiotemporal and event data in elite soccer. Scientific Data **12**(1), 195 (2025)

[4] Bauer, P., Anzer, G., Shaw, L.: Putting team formations in association football into context. Journal of sports analytics **9**(1), 39–59 (2023)

[5] Bekkers, J.: Pressing intensity: An intuitive measure for pressing in soccer. arXiv preprint arXiv:2501.04712 (2024)

[6] Bekkers, J.: EFPI: Elastic Formation and Position Identification in Football (Soccer) using Template Matching and Linear Assignment (June 2025), https://arxiv.org/abs/2506.23843, arXiv preprint arXiv:2506.23843

[7] Bekkers, J.: Wide open gazes: Quantifying visual exploratory behavior in soccer with pose enhanced positional data (2025). https://doi.org/10.6084/m9.figshare.29468036

[8] Bekkers, J., Dabadghao, S.: Flow motifs in soccer: What can passing behavior tell us? Journal of Sports Analytics **5**(4), 299–311 (2019)

[9] Bekkers, J., Sahasrabudhe, A.: A graph neural network deep-dive into successful counterattacks. arXiv preprint arXiv:2411.17450 (2024)

[10] Cao, Z., Simon, T., Wei, S.E., Sheikh, Y.: Realtime multi-person 2d pose estimation using part affinity fields. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 7291–7299 (2017)

[11] Casanova, F., Garganta, J., Silva, G., Alves, A., Oliveira, J., Williams, A.M.: Effects of prolonged intermittent exercise on perceptual-cognitive processes. Medicine & Science in Sports & Exercise **45**(8), 1610–1617 (2013)

[12] Caso, S., McGuckian, T.B., van der Kamp, J.: No evidence that visual exploratory activity distinguishes the super elite from elite football players. Science and Medicine in Football pp. 1–9 (2024)

[13] Chalkley, D., Shepherd, J.B., McGuckian, T.B., Pepping, G.J.: Development and validation of a sensor-based algorithm for detecting the visual exploratory actions. IEEE Sensors Letters **2**(2), 1–4 (2018)

[14] Cheng, B., Xiao, B., Wang, J., Shi, H., Huang, T.S., Zhang, L.: Higherhrnet: Scale-aware representation learning for bottom-up human pose estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5386–5395 (2020)

[15] Decroos, T., Bransen, L., Van Haaren, J., Davis, J.: Actions speak louder than goals: Valuing player actions in soccer. In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp. 1851–1861. KDD '19, ACM, New York, NY, USA (2019). https://doi.org/10.1145/3292500.3330758

[16] Eldridge, D., Pulling, C., Robins, M.T.: Visual exploratory activity and resultant behavioural analysis of youth midfield soccer players. Journal of Human Sport and Exercise **8**(3), 560–577 (2013)

[17] Euler, L.: Introductio in analysin infinitorum, vol. 2. MM Bousquet (1748)

[18] Fang, Y., Nakashima, R., Matsumiya, K., Kuriki, I., Shioiri, S.: Eye-head coordination for visual cognitive processing. PloS one **10**(3), e0121035 (2015)

[19] Fernández, J., Bornn, L.: Soccermap: A deep learning architecture for visually-interpretable analysis in soccer. In: Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 491-506 (2020)

[20] Fernández, J., Bornn, L.: Wide open spaces: A statistical technique for measuring space creation in professional soccer. In: Sloan sports analytics conference. vol. 2018 (2018)

[21] Henson, D.: Visual Fields. Oxford University Press, Oxford (1993)

[22] Jordet, G.: Applied cognitive sport psychology in team ball sports: an ecological approach. New Approaches to Sport and Exercise Psychology, eds R. Stelter and KK Roessler (Aachen: Meyer & Meyer Sport) pp. 147–174 (2005)

[23] Jordet, G., Bloomfield, J., Heijmerikx, J.: The hidden foundation of field vision in english premier league (epl) soccer players. In: Proceedings of the MIT sloan sports analytics conference. pp. 1–2 (2013)

[24] Kredel, R., Hernandez, J., Hossner, E.J., Zahno, S.: Eye-tracking technology and the dynamics of natural gaze behavior in sports: an update 2016–2022. Frontiers in Psychology **14**, 1130051 (2023)

[25] Lee, M., Jo, G., Hong, M., Bauer, P., Ko, S.K.: express: Contextual valuation of individual players within pressing situations in soccer. In: Proceedings of the MIT Sloan Sports Analytics Conference. Boston, MA (March 2025)

[26] Maas, T.R.: Monitoring of Visual Exploratory Activity in Professional Football Using a Camera-Based Detection Algorithm. Master's thesis, Eindhoven University of Technology (2025)

[27] McGuckian, T.B., Cole, M.H., Chalkley, D., Jordet, G., Pepping, G.J.: Constraints on visual exploration of youth football players during 11v11 match-play: The influence of playing role, pitch position and phase of play. Journal of Sports Sciences **38**(6), 658–668 (2020)

[28] McGuckian, T. B., Cole, M. H., Jordet, G., Chalkley, D., & Pepping, G. J.: Don't turn blind! The relationship between exploration before ball possession and on-ball performance in association football. Frontiers in psychology, 9, 2520 (2018)

[29] McGuckian, T.B., Cole, M.H., Pepping, G.J.: A systematic review of the technology-based assessment of visual perception and exploration behaviour in association football. Journal of Sports Sciences **36**(8), 861–880 (2018)

[30] Panchuk, D., Vickers, J.N.: Gaze behaviors of goaltenders under spatial–temporal constraints. Human Movement Science **25**(6), 733–752 (2006)

[31] Pokolm, M., Kirchhain, M., Müller, D., Jordet, G., Memmert, D.: Head movement direction in football-a field study on visual scanning activity during the uefa-u17 and -u21 european championship 2019. Journal of Sports Sciences **41**(7), 695–705 (2023)

[32] Power, P., Ruiz, H., Wei, X., Lucey, P.: Not all passes are created equal: Objectively measuring the risk and reward of passes in soccer from tracking data. In: Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining. pp. 1605–1613 (2017)

[33] Rahimian, P., Van Haaren, J., Abzhanova, T., Toka, L.: Beyond action valuation: A deep reinforcement learning framework for optimizing player decisions in soccer. In: 16th MIT sloan sports analytics conference. vol. 3 (2022)

[34] ReSpo.Vision: ReSpo.Vision (2025), https://respo.vision/

[35] Robberechts, P., Van Roy, M., Davis, J.: un-xpass: Measuring soccer player's creativity. In: Proceedings of the 29th ACM SIGKDD conference on knowledge discovery and data mining. pp. 4768–4777 (2023)

[36] Roca, A., Ford, P.R., McRobert, A.P., Williams, A.M.: Identifying the processes underpinning anticipation and decision-making in a dynamic time-constrained task. Cognitive Processing **12**(3), 301–310 (2011)

[37] Spearman, W., Basye, A., Dick, G., Hotovy, R., Pop, P.: Physics-based modeling of pass probabilities in soccer. In: Proceeding of the 11th MIT Sloan Sports Analytics Conference. vol. 1 (2017)

[38] Stats Perform: Stats Perform (2025), https://www.statsperform.com/

[39] Sun, K., Xiao, B., Liu, D., Wang, J.: Deep high-resolution representation learning for human pose estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5693–5703 (2019)

[40] Taki, T., Hasegawa, J.i.: Visualization of dominant region in team games and its application to teamwork analysis. Computer Graphics International 2000. Proceedings pp. 227–235 (2000)

[41] Teranishi, M., Tsutsui, K., Takeda, K., Fujii, K.: Evaluation of creating scoring opportunities for teammates in soccer via trajectory prediction. In: International Workshop on Machine Learning and Data Mining for Sports Analytics. pp. 53–73. Springer (2022)

[42] Vaeyens, R., Lenoir, M., Williams, A.M., Mazyn, L., Philippaerts, R.M.: Mechanisms underpinning successful decision making in skilled youth soccer players: An analysis of visual search behaviors. Journal of Motor Behavior **39**(5), 395–408 (2007)

[43] Van Roy, M., Cascioli, L., Davis, J.: Etsy: A rule-based approach to event and tracking data synchronization. In: Machine Learning and Data Mining for Sports Analytics ECML/PKDD 2023 Workshop. pp. 11,23. Springer (2023)

[44] Vossen, K.: Kloppy: standardizing soccer tracking and event data [github] (2020), https://github.com/PySport/kloppy

## Apéndice A. Definiciones de Símbolos

Una lista completa de símbolos en orden de primera aparición.

| Símbolo | Definición | Sección |
|---|---|---|
| p_i | Ubicación del jugador i | 2.2 |
| p_j | Ubicación del jugador j | 2.2 |
| p_b | Ubicación del balón | 2.2 |
| θ_h | Ángulo de rotación de cabeza | 2.2 |
| θ_s | Ángulo de rotación de hombros | 2.2 |
| ⊙ | Operador de multiplicación elemento a elemento | 3.1 |
| t | Tiempo actual | 3.1 |
| V_β | Campo de visión binario para el jugador i | 3.1 |
| V_ρ | Campo de visión probabilístico para el jugador i | 3.1 |
| v_i | Velocidad del jugador i | 3.1 |
| R(c_r(v_i)) | Componente de visión radial influenciado por la velocidad del jugador | 3.1 |
| A(c_a(v_i)) | Componente de visión angular influenciado por la velocidad del jugador | 3.1 |
| c_r | Parámetro de escalado que controla la tasa de decaimiento de la visión radial | 3.1 |
| d | Distancia desde el jugador i | 3.1 |
| σ_r | Desviación estándar de la profundidad de visión | 3.1 |
| c_a | Parámetro de escalado que controla la tasa de decaimiento de la visión angular | 3.1 |
| θ_a | Ángulo desde el punto focal | 3.1 |
| σ_a | Desviación estándar de la visión angular | 3.1 |
| V_Φ | Mapa de oclusión combinado para el jugador i | 3.1 |
| V_ϕ,i,j | Mapa de oclusión individual del jugador j tal como lo percibe el jugador i | 3.1 |
| J | Conjunto de todos los demás jugadores | 3.1 |
| V_o,i,j | Máscara binaria, la vista no obstruida entre los jugadores i y j | 3.1 |
| Q_i,j | Rayo probabilístico proyectado desde el jugador i a través del jugador j | 3.1 |
| α | Parámetro de probabilidad máxima de obstrucción | 3.1 |
| θ_q | Parámetro angular en la fórmula del rayo de oclusión | 3.1 |
| σ_q | Desviación estándar para el rayo de oclusión | 3.1 |
| c_q | Parámetro de escalado para el ancho del rayo de oclusión | 3.1 |
| δ_i,j | Distancia entre los jugadores i y j | 3.1 |
| ω_α | Ancho angular aparente del jugador j tal como lo percibe el jugador i | 3.1 |
| V | Mapa de visión completo del jugador i | 3.1 |
| PC | Control del campo | 3.2 |
| c_in | Parámetro de escalado para el radio de influencia del jugador R_i | 3.2 |
| PC^(c_in) | Control de campo inminente con parámetro de escalado c_in | 3.2 |
| J_att | Conjunto de jugadores del equipo atacante | 3.2 |
| J_def | Conjunto de jugadores del equipo defensor | 3.2 |
| PC^0.5_J_att | Control de campo inminente del equipo atacante excluyendo al jugador i | 3.2 |
| PC^0.5_J_def | Control de campo inminente del equipo defensor | 3.2 |
| PC^0.5_i | Control de campo inminente del jugador atacante i | 3.2 |
| R_i | Radio de influencia del jugador | 3.2 |
| V_l | Superficie de valor del campo dada la ubicación del balón | 3.2 |
| V_η | Superficie de normalización de distancia a la portería | 3.2 |
| V̂_l | Superficie de valor del campo normalizada | 3.2 |
| t^inicio_espera | Tiempo de inicio de la fase de espera | 3.2 |
| t^fin_espera | Tiempo de fin de la fase de espera | 3.2 |
| t^fin_con_balón | Tiempo de fin de la fase con balón | 3.2 |
| V_rat | Razón de espacio ocupado observado respecto al espacio ocupado total | 4 |
| P_v(t) | Valor instantáneo del campo del espacio controlado por el jugador i en el tiempo t | 4 |
| p_rat | Razón de aumento/disminución del valor del campo | 4 |
| y | Etiqueta binaria de disminución (0) o aumento (1) del valor del campo | 4 |
| ω_s | Ancho de hombros asumido de un jugador (0,5m) | C.2 |
| d_s | Profundidad de torso asumida de un jugador (0,3m) | C.2 |
| c^local_k | Posición local del torso de la esquina k | C.2 |
| R(θ_s) | Matriz de rotación para la orientación del jugador | C.2 |
| c_k | Posición global del torso de la esquina k | C.2 |
| v_k | Vector desde el jugador i a la esquina k del cuerpo del jugador j | C.2 |
| θ_a,b | Ángulo entre los vectores a y b | C.2 |
| θ_1,3 | Ángulo entre las esquinas opuestas 1 y 3 del torso del jugador j | C.2 |
| θ_2,4 | Ángulo entre las esquinas opuestas 2 y 4 del torso del jugador j | C.2 |

## Apéndice B. Factores de Escalado

### Apéndice B.1. Factores de Escalado del Mapa de Visión Periférica c_a y c_r

Definimos dos funciones lineales, determinadas empíricamente, para traducir la velocidad de un jugador (v_i) en el factor de escalado c_a siguiendo la Fórmula 13, y c_r siguiendo la Fórmula 14.

$$c_a = \min(0.3 \, v_i + 0.2, \, 0.5) \quad (13)$$

$$c_r = \min(0.25 \, v_i + 0.1, \, 2.6) \quad (14)$$

### Apéndice B.2. Factor de Escalado del Mapa de Oclusión c_q

**Ancho Aparente (ω_α)**

El ancho aparente (ω_α) es un ángulo (en radianes) que representa el ancho angular del cuerpo de un jugador *j* - representado como un rectángulo - tal como lo ve el jugador *i* en el plano bidimensional cenital. Dada la posición p_i = [x_i, y_i] y la posición p_j = [x_j, y_j], donde el jugador *j* se representa como un rectángulo con ancho ω_s, profundidad d_s, y orientación θ_s, calculamos el ancho aparente del cuerpo observado del jugador *j* calculando los cuatro puntos (k) de su torso rectangular (en el plano 2D) en espacio local con la Fórmula 15a, creando la matriz de rotación para la orientación del rectángulo usando la rotación de hombros del jugador *j* (θ_s) con la Fórmula 15b, y luego calculando la posición global del torso (c_k) para los cuatro puntos (k) mediante la Fórmula 15c.

$$c^{local}_k = \left\{ \begin{bmatrix} \frac{ω_s}{2} \\ \frac{d_s}{2} \end{bmatrix}, \begin{bmatrix} -\frac{ω_s}{2} \\ \frac{d_s}{2} \end{bmatrix}, \begin{bmatrix} -\frac{ω_s}{2} \\ -\frac{d_s}{2} \end{bmatrix}, \begin{bmatrix} \frac{ω_s}{2} \\ -\frac{d_s}{2} \end{bmatrix} \right\} \quad (15a)$$

$$R(θ_s) = \begin{bmatrix} \cos(θ_s) & -\sin(θ_s) \\ \sin(θ_s) & \cos(θ_s) \end{bmatrix} \quad (15b)$$

$$c_k = R(θ_s) \cdot c^{local}_k + p_j, \quad k \in \{1,2,3,4\} \quad (15c)$$

Posteriormente, calculamos los vectores (v_k) desde la posición del jugador *i* (p_i) a cada esquina del cuerpo del jugador *j* (c_k) con la Fórmula 15d. Entonces, el ancho del cuerpo del jugador *j* tal como lo percibe el jugador *i* en el plano bidimensional es el ángulo entre las líneas de visión a las esquinas opuestas del torso del jugador *j* calculado en la Fórmula 15e. Aquí el ángulo (θ) entre los vectores v_a y v_b se calcula para las dos esquinas opuestas [(v_1, v_3) y (v_2, v_4)] del torso del jugador *j*. Finalmente, el ancho aparente del cuerpo del jugador *j* visto desde la perspectiva del jugador *i*, en radianes, es el ángulo máximo entre cualquier par de esquinas opuestas (Fórmula 15f).

$$v_k = c_k - p_i, \quad k \in \{1,2,3,4\} \quad (15d)$$

$$θ_{a,b} = \arccos\left(\frac{v_a \cdot v_b}{\|v_a\| \|v_b\|}\right) \quad (15e)$$

$$ω_α = \max(θ_{1,3}, \, θ_{2,4}) \quad (15f)$$

**Distancia Entre el Jugador *i* y *j* (δ_i,j)**

La distancia entre el jugador *i* y el jugador *j*, utilizada para escalar apropiadamente ω_α relativo a las posiciones de ambos jugadores se calcula siguiendo la Fórmula 16.

$$δ_{i,j} = \|p_j - p_i\| \quad (16)$$

## Apéndice C. Distancia a la Portería del Oponente (Línea)

A pesar de su aparente similitud, "Distancia a la Línea de Gol del Oponente" y "Distancia a la Portería del Oponente" ejercen efectos opuestos en los valores predichos. Esta aparente contradicción se resuelve considerando que los jugadores posicionados a distancias intermedias de la línea de gol pero más cerca de la portería controlan mayor valor de campo inminente que aquellos cerca de la línea de gol pero lejos de la portería (p. ej., cerca de las banderas de esquina). Esto se refuerza aún más por el modelo de valor del campo (como se muestra en la Figura 11) que muestra que las áreas amplias son significativamente menos valiosas, y por la escasez inherente de muestras donde un jugador espera un pase muy cerca de la portería y muy cerca de la línea de gol (p. ej., en el área pequeña) y posteriormente ejecuta un movimiento con balón. Nuestro enfoque agresivo de filtrado como se discute en la Sección 4.2 refuerza esto aún más porque elimina más casos extremos.
