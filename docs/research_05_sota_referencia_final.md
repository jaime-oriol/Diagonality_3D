# Diagonalidad computacional en futbol: documento tecnico de referencia para TFG

**La interseccion entre direccion de accion ofensiva, disrupcion defensiva e inferencia causal constituye una laguna de investigacion confirmada.** Tras una busqueda exhaustiva en arXiv, Google Scholar, Springer, MIT Sloan, JQAS, Journal of Sports Sciences, Frontiers, IEEE, PubMed y DMKD, se confirma que (1) ningun paper ha correlacionado la direccion/angulo de una accion con los componentes direccionales de la disrupcion defensiva, (2) no existe extension del PPCF que dependa de la direccion de llegada del balon, y (3) no se ha aplicado DML con tratamiento continuo en football analytics. Estas tres lagunas definen el espacio de contribucion original del TFG. El presente documento recoge el estado del arte completo, herramientas, decisiones tecnicas, pipeline y propuestas de titulo para servir como referencia unica durante todo el desarrollo.

---

## BLOQUE 1 — ESTADO DEL ARTE EXHAUSTIVO

---

### 1. Disrupcion defensiva: de D-Def a la laguna direccional

**Goes, Kempe, Meerhoff y Lemmink (2019)** introdujeron D-Def en *Big Data* (DOI: 10.1089/big.2018.0067). El vector de estado defensivo tiene **10 variables**: centroide de equipo (Cx, Cy), centroides de linea defensiva (Cx_def, Cy_def), linea media (Cx_mid, Cy_mid) y linea atacante (Cx_att, Cy_att), superficie del convex hull (S_area) y spread (norma de Frobenius de la matriz de posiciones). La asignacion a lineas se realiza via **K-Means (k=3)** sobre posiciones medias del primer tiempo. Los cambios en las 10 variables se computan en una **ventana fija de 3 segundos** tras el pase, se estandarizan con Z-score, y se reducen mediante **PCA a 3 componentes que explican el 83.3% de la varianza**: PC1 = disrupcion longitudinal, PC2 = disrupcion lateral, PC3 = disrupcion de forma (superficie + spread). La formula final es **D-Def = |PC1| + |PC2| + |PC3|**, con rango 0-150. Limitacion clave: la ventana de 3 s es fija, la deteccion de lineas es estatica y el portero se excluye.

**Forcher, Altmann, Jekauc y Kempe (2021)** validaron D-Def en *Entropy* (DOI: 10.3390/e23121607) con **258 partidos de la Eredivisie 2018/19** (13 094 ataques deliberados). Analizaron los ultimos 4 pases de cada ataque. Los resultados cuantitativos mas relevantes: D-Def del penultimo pase ("hockey assist") y D-Def maximo del ataque mostraron los tamanios de efecto mas altos con **d de Cohen = 0.23** (p < 0.001). Los ataques exitosos requieren al menos un pase con D-Def > 28. Hallazgo critico para el TFG: los angulos de pase difirieron significativamente entre ataques exitosos e infructuosos (Watson-Williams F-test, p < 0.001 en los 4 pases), con **mas pases diagonales en ataques exitosos**, pero los autores no correlacionaron el angulo de pase con los componentes direccionales de D-Def (PC1/PC2). Esta es precisamente la laguna que el TFG propone cubrir.

**Forcher et al. (2024)** publicaron en *IJSSC* (DOI: 10.1177/17479541231172695) el analisis de 153 partidos de Bundesliga 2020/21, comparando compacidad local vs global. Hallazgo principal: la compacidad global del equipo **no discrimina** entre defensas exitosas y fallidas, pero la **compacidad local** (subgrupo de 5 defensores mas cercanos al balon) si lo hace (d entre -0.08 y -0.16). Distancia inter-lineas: defensa-medio = 10.22 +/- 3.76 m; medio-ataque = 13.30 +/- 4.22 m.

**Frencken, Lemmink, Delleman y Visscher (2011)** definieron en *EJSS* (DOI: 10.1080/17461391.2010.499967) las metricas fundacionales: centroide (media X,Y de jugadores), superficie (area del convex hull) y stretch index (distancia radial media al centroide). En juegos reducidos 4v4, la correlacion longitudinal de centroides entre equipos fue r > 0.94 (comportamiento en fase).

**Goes et al. (2021)** cuantificaron en *J Sports Sciences* (DOI: 10.1080/02640414.2020.1834689) la sincronia inter-equipo con **transformada de Hilbert** sobre series temporales de centroides de subgrupos (118 partidos Eredivisie, 12 424 ataques). Resultado clave: los ataques exitosos mostraron **desincronizacion longitudinal** significativa (p < 0.01) entre defensores del equipo atacante y defensores rivales. Las variables de subgrupo fueron mas sensibles que las de equipo completo.

**Link, Lang y Seidenschwarz (2016)** definieron Dangerousity en *PLOS ONE* (DOI: 10.1371/journal.pone.0168768) como probabilidad en tiempo real de gol, con 4 componentes: Zone (posicion del IBA), Control (dinamica del balon), Pressure (defensores cercanos) y Density (ratio atacantes/defensores en zona de intercepcion). Validado con 100 escenarios por 3 entrenadores semi-profesionales (F = 170.31, p < 0.01). A diferencia de D-Def, opera solo en los 34 m finales y valora inherentemente la proximidad a porteria.

**La distancia inter-lineas** no tiene un paper fundacional unico. Es un concepto emergente del coaching que se formalizo operativamente en Goes et al. (2019, 2021) como distancia longitudinal entre centroides de lineas asignadas via clustering, y en Forcher et al. (2024) como variable organizativa explicita.

**Laguna confirmada sobre direccion:** Tras busqueda exhaustiva, **ningun estudio publicado** ha correlacionado sistematicamente la direccion/angulo de un pase con el componente direccional especifico (PC1 longitudinal vs PC2 lateral) de la disrupcion defensiva resultante. D-Def descompone la disrupcion en ejes, pero ninguna publicacion analiza que angulos de pase maximizan que componente. Forcher et al. (2021) muestran que los pases diagonales son mas frecuentes en ataques exitosos, pero no vinculan angulo con componente de D-Def. Este es el gap central del componente A del TFG.

---

### 2. Pitch Control: del PPCF de Spearman a la ausencia de extension direccional

**Spearman (2017)** presento en MIT Sloan "Physics-Based Modeling of Pass Probabilities", introduciendo el campo de control como probabilidad de que un equipo controle el balon en cada punto. Modelo de movimiento: el jugador continua a velocidad actual durante t_react, luego acelera a v_max hacia el objetivo. Probabilidad de intercepcion modelada con **funcion logistica** (colas mas pesadas que normal). Precision: 81% equipo receptor, 68% receptor especifico.

**Spearman (2018)** extendio el modelo a **PPCF (Probabilistic Pitch Control Function)** en "Beyond Expected Goals" (MIT Sloan). Parametros exactos de la implementacion de referencia (Laurie Shaw): **t_react = 0.7 s**, **v_max = 5 m/s**, **sigma_tti = 0.45 s** (desviacion estandar logistica del tiempo de intercepcion), **kappa_def = 1.72** (factor de escala: defensores necesitan menos tiempo para "controlar"), velocidad media del balon = 15 m/s, paso de integracion dt = 0.04 s, tiempo maximo de integracion = 10 s, tolerancia de convergencia = 0.01. La ecuacion diferencial del PPCF integra contribuciones de todos los jugadores: dPPCF/dT = (1 - PPCF_att - PPCF_def) x P_intercept(T) x lambda.

**Fernandez y Bornn (2018)** propusieron en MIT Sloan ("Wide Open Spaces") un modelo alternativo donde la influencia de cada jugador es una **gaussiana bivariante rotada por la direccion de su velocidad**: Sigma_i = R(theta) x S x S^T x R(theta)^T, con la gaussiana elongada en la direccion del movimiento. El radio de influencia varia entre **4 y 10 metros** en funcion de la distancia al balon. El pitch control a nivel de equipo se obtiene via funcion logistica sobre la diferencia de influencias.

**Efthimiou (2021, 2023)** propuso en arXiv (arXiv: 2107.05714) y en FICC 2023 (DOI: 10.1007/978-3-031-28076-4_37) reemplazar Voronoi estandar por **diagramas de Apolonio** (Voronoi generalizado para velocidades distintas) y extendio el modelo con asimetria direccional: "los jugadores tienen mas control en la direccion en que corren que en cualquier otra direccion." Las regiones de dominancia resultantes pueden ser **no convexas, contener agujeros o estar desconectadas**.

**Bekkers (2025)** publico "Pressing Intensity" (arXiv: 2501.04712), un framework que reutiliza componentes del modelo de Spearman (TTI con velocidad y reaccion) para cuantificar presion defensiva. La intensidad de pressing sobre el jugador j se computa como **1 - prod(1 - p_{i,j})** para los 11 defensores, donde p_{i,j} es la probabilidad de intercepcion dentro de 1.5 s. Implementado en el paquete open-source `unravelsports`.

**SoccerMap (Fernandez y Bornn, 2020)** en ECML-PKDD (arXiv: 2010.10202, DOI: 10.1007/978-3-030-67670-4_30) introdujo una **CNN fully-convolutional** que estima superficies completas de probabilidad de pase desde datos de tracking renderizados como capas espaciales multi-canal. Innovacion: Target-Location Loss para aprender superficies completas desde ground-truth puntual.

**Overmeer, Janssen y Nuijten (2025)** publicaron en arXiv (arXiv: 2502.02565, DOI: 10.5220/0013784300003988) una arquitectura **U-Net con skip connections** para EPV, un benchmark cuantitativo (OJN-Pass-EPV), la inclusion de **altura del balon** como feature, y un modelo dual reward/risk para pases. Precision del benchmark: 78% en pares de estados.

**Laguna confirmada sobre PPCF direccional:** Tras busqueda exhaustiva (2016-2026), **no existe ninguna extension publicada del PPCF que haga depender la probabilidad de control de la direccion de llegada del balon a un punto**. El PPCF de Spearman incorpora el tiempo de vuelo del balon (implicitamente la posicion del pasador), pero no modela como el angulo de llegada afecta la capacidad de control (un balon llegando por detras es mas dificil de controlar que uno frontal). Fernandez y Bornn rotan la gaussiana por velocidad del jugador, no por direccion del balon. Efthimiou modela asimetria por direccion de carrera del jugador. Ninguno modela la direccion del pase entrante. Esta laguna define el componente B del TFG: Diagonal Opportunity Surfaces (DOS).

---

### 3. Angulos y direccion en football analytics

**Cordon-Carmona, Garcia-Aliaga, Marquina, Calvo, Mon-Lopez y Refoyo Roman (2020)** publicaron en *IJERPH* (DOI: 10.3390/ijerph17249396) un analisis observacional de 20 partidos de La Liga 2018/19 donde las carreras diagonales del receptor mostraron asociacion moderada con el exito del ataque (chi-cuadrado, p < 0.05), con **aproximadamente +7% de tasa de exito** respecto a movimientos rectilíneos. Nota: el usuario referencia "Gonzalez-Rodenas et al." pero la descripcion coincide con este paper de Cordon-Carmona et al.; existe tambien Gonzalez-Rodenas et al. (2020) en PLOS ONE (DOI: 10.1371/journal.pone.0226978) sobre efectividad ofensiva en Premier League con enfasis en contraataques.

**Anzer y Bauer (2022)** presentaron xPass en *DMKD* (DOI: 10.1007/s10618-021-00810-3), un modelo que usa los primeros **0.4 s de trayectoria del balon** y vectores de movimiento de los 22 jugadores para predecir el receptor intencionado (93.0% precision en pases exitosos). La direccion se codifica implicitamente a traves de la trayectoria del balon, posiciones relativas y movimientos de jugadores. SHAP values confirmaron que las features relacionadas con la direccion del pase estan entre las mas influyentes.

**Arbues-Sanguesa, Martin, Fernandez, Haro y Ballester (2020)** propusieron en CVPR Workshop (arXiv: 2004.07209) un modelo de factibilidad geometrica de pases basado en la **orientacion corporal** del jugador ofensivo. Con orientacion como medida de factibilidad, logran > 0.7 Top-3 accuracy en prediccion de pases. El paper de 2021 (arXiv: 2106.00359) introduce el primer modelo de deep learning para estimar orientacion corporal desde video, usando bins de clasificacion con loss ciclica y ground-truth de dispositivos EPTS. La orientacion en "half-open body" facilita significativamente los pases progresivos.

**Morishita, Aruga, Nakayama, Kijima y Shima (2025)** aplicaron en *Physica A* (DOI: 10.1016/j.physa.2025.130507) la **descomposicion de Helmholtz** (potencial escalar + potencial vectorial) a campos vectoriales de ultimos pases. El potencial escalar identifica zonas de convergencia de pases (minimo frente al area, ligeramente a la izquierda, reflejando cruces desde la derecha). El potencial vectorial (componente rotacional) revela patrones tacticos de respuesta a defensores en zonas centrales y laterales. Enfoque novedoso que conecta fisica de campos con tactica futbolistica.

**Herold, Hecksteden, Radke, Goes, Nopp, Meyer y Kempe (2022)** publicaron en *J Sports Sciences* (DOI: 10.1080/02640414.2022.2081405) un modelo de presion defensiva eliptica para evaluar acciones off-ball de alta intensidad: 988 Deep Runs y 423 Changes of Direction (CODs) en 22 partidos de la seleccion alemana. Ambos tipos reducen la presion defensiva sobre el receptor, pero los CODs (cambios de direccion, frecuentemente diagonales) son particularmente efectivos.

**Cho, Ryu y Song (2022)** introdujeron Pass2vec en *IJSSC* (DOI: 10.1177/17479541211033078), un autoencoder convolucional que combina localizacion, longitud y **direccion de pase** en embeddings a nivel de jugador. Precision en retrieval de jugadores: 76.5% Top-20.

**Papers recientes 2024-2026:** Morishita et al. (2025) es el mas relevante para analisis vectorial de direccion. La narrativa tactica sobre "diagonalidad" ha emergido fuertemente en 2025 (Spielverlagerung: Rafelt y Maric; The Football Analyst; Tactics Journal) definiendo diagonalidad como "conectar progresion vertical con seguridad horizontal", pero sin formalizacion cuantitativa en papers academicos. Este hueco tactica-analitica es central para el TFG.

---

### 4. Inferencia causal en sports analytics: del DML teorico a la aplicacion incipiente

**Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey y Robins (2018)** establecieron el framework DML en *The Econometrics Journal* (DOI: 10.1111/ectj.12097). Procedimiento en dos etapas: (1) ML para estimar parametros nuisance (modelo de resultado y modelo de tratamiento); (2) cross-fitting y funciones de score Neyman-ortogonales para desbiesar las estimaciones. La ortogonalizacion elimina el sesgo de regularizacion de los estimadores ML de primera etapa.

**Colangelo y Lee (2020/2025)** extendieron DML a **tratamientos continuos** (arXiv: 2004.03036, publicado en *JBES*, DOI: 10.1080/07350015.2025.2505487). Estiman la **funcion de dosis-respuesta promedio** beta(t) = E[Y(t)] y efectos parciales del_beta(t)/del_t usando un momento doblemente robusto con kernel y cross-fitting. Tres pasos: (1) cross-fitting en L folds, (2) estimacion doblemente debiased con kernel alrededor del valor de tratamiento t, (3) efectos parciales via diferenciacion finita. Las funciones nuisance son gamma(t,x) = E[Y|T=t,X=x] y f_{T|X}(t|x) (GPS). Proveen condiciones suficientes para estimadores kernel, series y redes neuronales profundas. Este es el paper metodologico central para el componente A del TFG.

**Hirano e Imbens (2004)** definieron el GPS (Generalized Propensity Score) como la densidad condicional del tratamiento dadas las covariables: r(t,x) = f_{T|X}(t|x). Bajo unconfoundedness debil, ajustar por el GPS elimina el sesgo de confusores. Implementacion clasica: regresion incluyendo GPS como covariable.

**Athey, Tibshirani y Wager (2019)** formalizaron los Generalized Random Forests en *Annals of Statistics* (arXiv: 1610.01271). Extienden random forests para estimar CATE (Conditional Average Treatment Effect) con estimacion honesta, centrado local e intervalos de confianza asintoticos. Implementado en `grf` (R).

**VanderWeele y Ding (2017)** introdujeron E-values en *Annals of Internal Medicine* (DOI: 10.7326/M16-2607). El E-value cuantifica la fuerza minima de confounding no medido necesaria para explicar una asociacion observada: E-value = RR + sqrt(RR x (RR - 1)). Si el E-value es grande relativo a confounders conocidos, el resultado es robusto.

**Alam, Moodie, Wu y Swartz (2025)** publicaron en arXiv (arXiv: 2505.11841) un caso tutorial de inferencia causal para cruces en futbol: ATE vs ATT via propensity score matching con datos de tracking de Shandong Taishan Luneng FC (2017 CSL, 2225 oportunidades de cruce). ATE = +1.6% probabilidad de tiro; ATT = +5.0%. El ATT mayor sugiere autoseleccion hacia contextos favorables.

**Aplicaciones de DML en futbol (confirmadas):**

- **Bajons (2023)** en ACM ICoMS (DOI: 10.1145/3613347.3613368): DML para estimar contribuciones no sesgadas de jugadores a posesiones, derivando ratings individuales.
- **Ruiz-Menarguez y Badiella (2026)** en arXiv (arXiv: 2602.16830): DML adaptado a **tratamientos categoricos** (formaciones) con residualizacion matricial y XGBoost. 22 000+ partidos de ligas europeas. La 4-2-3-1 vs 3-5-2 muestra ventaja estimada de 0.16 goles. Unico paper encontrado que aplica DML directamente en futbol con formaciones como tratamiento.

**Otras aplicaciones causales en futbol:** Dona y Swartz (2024, *IMA J Management Mathematics*) para saques de banda; Ali y Yilmaz (2023, *IEEE Access*, DOI: 10.1109/ACCESS.2023.3333878) para sustituciones con ML causal; Klemp (2024/2025, tesis doctoral DSHS Koln) sobre la importancia del contexto causal en match analysis research.

**Lagunas confirmadas:** (1) **Ningun paper ha aplicado DML con tratamiento continuo en football analytics.** (2) **Ningun paper ha usado inferencia causal para evaluar el efecto de la direccion de accion en deporte.** (3) **No existen curvas dosis-respuesta publicadas en sports analytics.** Wu y Chen (2025, arXiv: 2507.19889) desarrollaron inferencia causal para datos circulares (angulos) definiendo efectos tratamiento como diferencias en vectores resultantes, pero no en contexto deportivo. La aplicacion de DML continuo (Colangelo y Lee) al angulo de pase como tratamiento continuo seria metodologicamente original.

---

### 5. Geometria computacional en futbol

**Narizuka y Yamazaki (2019)** en *Scientific Reports* (DOI: 10.1038/s41598-019-48623-1) definieron la formacion en cada frame como la **matriz de adyacencia de la triangulacion de Delaunay** de las 10 posiciones de jugadores de campo. Introdujeron una medida de disimilitud entre formaciones basada en distancia euclidiana entre matrices de adyacencia, y aplican clustering jerarquico (metodo de Ward). Limitacion: solo usa la estructura topologica (adyacencia), no las propiedades geometricas de las aristas.

**Raabe et al. (2024)** en *J Sports Sciences* (DOI: 10.1080/02640414.2024.2414363) extendieron la perspectiva Voronoi inter-equipo con una aplicacion **intra-equipo** de Delaunay para medir gestion espacial. Analizaron 128 187 secuencias atacantes de 306 partidos de elite. Hallazgo clave: el exito atacante se caracteriza por **triangulos grandes cerca del balon**, no por el numero total de triangulos. Modelos mixtos lineales para validacion.

**Brandes, Sotudeh, Parlak, Laffranchi y Erkul (2025)** en *npj Complexity* (DOI: 10.1038/s44260-025-00047-x) introdujeron **shape graphs**: subgrafos de triangulaciones de Delaunay inspirados en reconocimiento de huellas dactilares y expresiones faciales. A diferencia de enfoques anteriores que agregan posiciones en el tiempo, interpretan **cada frame individualmente** a la maxima resolucion temporal. Manejan la inestabilidad angular eliminando preventivamente aristas con estabilidad angular baja. Introducen position plots que capturan la fluidez del posicionamiento relativo.

**Deteccion de lineas defensivas — Michalczyk / Stats Perform:** Utiliza **Jenks Natural Breaks** (Fisher-Jenks) con **3 clusters** sobre la coordenada x de jugadores de campo (promediada en ventana de 2 s para estabilidad; agrupaciones < 1 s eliminadas). Define "angle of view" como angulo maximo creado por el balon y dos jugadores adyacentes de la primera linea rival. Precision del modelo basado solo en eventos: 84% accuracy, 93% AUC, 89% recall. Bauer y Anzer (2021, *DMKD*) se centraron en deteccion de contrapresion, no en lineas defensivas en juego abierto.

**Convex hull defensivo:** Usado extensamente por Moura et al. (2012/2013), Shaw y Glickman (2019) (el convex hull atacante es el doble del defensivo), Castellano et al. (2013), Clemente et al. (2013). En 2025, *Scientific Reports* introdujo el **inner convex hull** y el layer ratio LR = area interna / area externa, con pico universal en LR aproximadamente 0.18 independientemente de la fase de juego.

**Laguna confirmada:** **Ningun paper publicado ha utilizado la orientacion de aristas de Delaunay como feature en football analytics.** Narizuka usa matrices de adyacencia (binarias), Raabe usa areas de triangulos, Brandes usa topologia de subgrafos. La orientacion de aristas SI se usa en otros dominios (minutiae de huellas, landmarks faciales), lo que precisamente inspira a Brandes et al., pero no se ha aplicado como feature tactica en futbol.

---

### 6. Tiempos de reaccion y biomecanica: hacia t_react(angulo)

**Vater (2024)** en *Scientific Reports* (DOI: 10.1038/s41598-024-53706-9) demostro en un CAVE con motion-tracking que el **tiempo de respuesta aumenta monotonicamente con el angulo de excentricidad visual** (hasta 90 grados izquierda/derecha) en defensa de baloncesto. Efecto de crowding confirmado. Diferencias de expertise solo en tareas complejas/representativas: los jugadores habilidosos compensan parcialmente el efecto de excentricidad en situaciones de juego real.

**Dos'Santos, Thomas, Comfort y Jones (2018)** revisaron en *Sports Medicine* (DOI: 10.1007/s40279-018-0968-3) el **trade-off angulo-velocidad** en cambios de direccion: velocidades de aproximacion mas rapidas comprometen la ejecucion del angulo de COD deseado. Los COD a **90 grados** muestran los mayores momentos de rotacion interna y abduccion de rodilla. A mayor angulo de COD: disminuye el perfil de velocidad, aumenta el tiempo de contacto con el suelo, disminuye GRF vertical. Existe un **conflicto rendimiento-lesion**: los COD mas agudos y rapidos son los mas eficaces tacticamente pero los de mayor riesgo de lesion.

**Mornieux, Gehring, Furst y Gollhofer (2014)** en *J Sports Sciences* (DOI: 10.1080/02640414.2013.876508) mostraron que con menos tiempo disponible antes de un corte: la cabeza esta menos orientada hacia la direccion de corte (p = 0.033), el tronco rota mas en la direccion opuesta (p = 0.002), y la mayor flexion lateral del tronco correlaciona con mayor momento de abduccion de rodilla (r = 0.41, p = 0.009). Los APAs (Anticipatory Postural Adjustments) son cruciales para el exito del corte.

**Ando, Kida y Oda (2001)** midieron en *Perceptual and Motor Skills* (DOI: 10.2466/pms.2001.92.3.786) el tiempo de reaccion visual central vs periferico con EMG: el RT periferico es mayor que el central, atribuible a **mayor tiempo premotor** (procesamiento SNC), no a tiempo motor. Los futbolistas mostraron tiempos premotores mas cortos tanto en vision central como periferica respecto a no atletas.

**Klostermann, Vater, Kredel y Hossner (2020)** propusieron en *Frontiers in Sports and Active Living* (DOI: 10.3389/fspor.2019.00066) un framework teorico sobre vision foveal vs periferica, distinguiendo tres estrategias: foveal spot, gaze anchor y visual pivot. Los expertos muestran estrategias superiores de gaze-anchoring que compensan parcialmente los efectos de excentricidad. Vater, Pinchuk y Vukojevi c (2026) confirmaron en un congreso que los futbolistas son afectados por angulos visuales pero no por crowding.

**Laguna confirmada: no existe una funcion parametrica publicada t_react(angulo).** La evidencia empirica soporta unanimemente que el RT crece monotonicamente con la excentricidad angular. Ando et al. (2001) usaron solo dos categorias (central vs periferico). Vater (2024) midio a angulos discretos hasta 90 grados pero no ajusto un modelo parametrico. La literatura psicofisica general documenta crecimiento aproximadamente logaritmico o lineal en ciertos rangos, pero ningun modelo estandar t_react(theta) ha sido publicado para contextos deportivos. Esto es una **oportunidad de modelado para el TFG**: proponer una funcion parametrica sencilla (e.g., logistica o lineal a trozos) calibrada con datos experimentales publicados para modular el parametro t_react del PPCF segun el angulo entre la orientacion del defensor y la direccion de llegada del balon.

---

## BLOQUE 2 — HERRAMIENTAS Y LIBRERIAS

| Herramienta | Version | Licencia | GitHub | Funcion principal |
|---|---|---|---|---|
| **unravelsports** | 1.2.1 | MPL-2.0 | github.com/UnravelSports/unravelsports | Pressing intensity, GNN, formaciones. Integra kloppy. |
| **LaurieOnTracking** | sin versionar (2022) | MIT | github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking | Implementacion de referencia de PPCF (Spearman), EPV, velocidades. |
| **Metrica-pitch-control** | sin versionar | no especificada | github.com/anenglishgoat/Metrica-pitch-control | PPCF en **PyTorch GPU**; partido completo en ~30 s. |
| **socceraction** | 1.5.3 | MIT | github.com/ML-KULeuven/socceraction | SPADL, VAEP, xT. Clientes para StatsBomb/Opta/Wyscout. |
| **kloppy** | 3.18.0 | BSD-3 | github.com/PySport/kloppy | Estandarizacion vendor-independiente de tracking/eventos. |
| **mplsoccer** | 1.6.1 | MIT | github.com/andrewRowlinson/mplsoccer | Visualizacion: pitch plots, heatmaps, Voronoi, convex hull. |
| **EconML** | 0.16.0 | MIT | github.com/py-why/EconML | DML, CausalForestDML, DRLearner, MetaLearners, DeepIV, SHAP. |
| **DoWhy** | 0.14 | MIT | github.com/py-why/dowhy | Flujo Model-Identify-Estimate-Refute; integracion con EconML. |
| **causal-curve** | 1.0.6 | MIT | github.com/ronikobrosly/causal-curve | GPS para tratamiento continuo; curvas dosis-respuesta con IC. |
| **grf** | 2.6.1 | GPL-3 | github.com/grf-labs/grf | Causal forests en R; estimacion honesta, IC jackknife. |
| **scipy.spatial.Delaunay** | SciPy >= 1.15 | BSD-3 | github.com/scipy/scipy | Triangulacion de Delaunay N-dimensional via Qhull. |
| **jenkspy** | 0.4.1 | MIT | github.com/mthh/jenkspy | Fisher-Jenks Natural Breaks; extension C para rendimiento. |

**Codigo open-source para D-Def:** No existe. La metrica necesita ser **reimplementada** desde las descripciones del paper (centroide, K-Means para lineas, PCA, ventana 3 s). La documentacion en Goes et al. (2019) es suficientemente detallada para reproducirla.

**Codigo para deteccion de lineas defensivas:** No existe como libreria dedicada. `unravelsports` incluye EFPI (Elastic Formation and Position Identification). La implementacion con `jenkspy` (3 clusters sobre coordenada x, ventana 2 s) siguiendo a Michalczyk es directa.

**Codigo para orientacion corporal desde tracking:** No hay codigo publico especifico. MEBOW (github.com/ChenyanWu/MEBOW, CVPR 2020) estima orientacion desde video pero no es futbol-especifico. El codigo de Arbues-Sanguesa no esta publicado. La heuristica comun desde tracking puro es estimar orientacion a partir del vector velocidad del jugador, que es una simplificacion pero ampliamente usada.

---

## BLOQUE 3 — DECISIONES TECNICAS

### D-Def: ventana temporal y vector de estado

La ventana de **3 segundos** de Goes et al. (2019) es el estandar, pero Forcher et al. (2021) no exploraron alternativas. Para el TFG se recomienda computar D-Def a **2, 3 y 4 s** y seleccionar la ventana que maximice la correlacion con el outcome (xG generado o si se crea ocasion). El vector S_t tiene 10 componentes. **PCA es preferible a features directas** porque (1) reduce colinealidad entre centroides de lineas adyacentes, (2) proporciona la descomposicion longitudinal/lateral/forma que es central para la hipotesis del TFG, y (3) facilita la interpretabilidad. Alternativa: usar PC1-PC3 como outcomes separados en tres modelos causales para identificar que componente de disrupcion es mas sensible al angulo de pase.

### D-PPCF: resolucion, frecuencia y direcciones

Para la extension direccional del PPCF (Diagonal Opportunity Surfaces), las decisiones criticas son:

- **Resolucion del grid:** 104 x 68 celdas (1 m x 1 m) es el estandar en SoccerMap y Overmeer et al. El PPCF de Spearman en la implementacion de Shaw usa grids mas gruesos (50 x 32 ~ 2 m). Para GPU con PyTorch (anenglishgoat), 104 x 68 es viable.
- **Frecuencia temporal:** El tracking suele ser a 25 Hz; PPCF se computa tipicamente a **10 Hz** (subsampling) o por evento. Para DOS, computar por frame de evento (pase recibido) es suficiente; el coste de anadir la dimension direccional se amortiza no necesitando computar todos los frames.
- **Numero de direcciones:** Discretizar la direccion de llegada en **N_theta = 8 o 12 sectores angulares** (cada 45 o 30 grados). 8 direcciones (N, NE, E, SE, S, SW, W, NW) proporcionan un balance entre granularidad y coste. El PPCF se computa independientemente para cada direccion, dando un tensor de salida de dimension (104 x 68 x N_theta). La superficie DOS(x,y) = max_theta PPCF(x,y,theta) - min_theta PPCF(x,y,theta) cuantifica la ganancia maxima por eleccion optima de direccion.
- **Modulacion de t_react por angulo:** Proponer t_react(theta) = t_base + delta_t x f(theta), donde theta es el angulo entre la orientacion del defensor y la direccion de llegada del balon, y f es una funcion monotona creciente (e.g., sigmoide centrada en 90 grados). Calibrar t_base = 0.7 s (Spearman), delta_t entre 0.1 y 0.5 s basado en datos de Vater (2024) y Ando et al. (2001).

### Confounders exhaustivos para el modelo causal

El modelo causal estima E[Y(t)] donde T = angulo de accion (tratamiento continuo) e Y = disrupcion defensiva. Los confounders esenciales son:

- **Espaciales:** posicion (x,y) del pase, distancia a porteria, zona del campo (categorica: build-up / middle third / final third), altura del ataque (distancia de la linea defensiva rival a su porteria).
- **Cineticos del pase:** velocidad del balon, longitud del pase.
- **Contexto defensivo:** numero de defensores entre balon y porteria, PPCF en el punto de recepcion, pressing intensity (Bekkers) sobre el pasador, compacidad local defensiva (Forcher et al. 2024).
- **Contexto ofensivo:** numero de companeros en 10 m del receptor, velocidad del receptor, orientacion del receptor (si disponible), fase de juego (posesion establecida vs transicion vs jugada a balon parado).
- **Temporales:** minuto de partido, marcador diferencial, contexto de partido (casa/fuera).
- **Secuenciales:** D-Def acumulado de los pases anteriores de la secuencia, numero de pases en la secuencia actual.

Para la validacion de robustez, **E-values** (VanderWeele y Ding, 2017) cuantifican cuanto confounding no medido seria necesario para invalidar los hallazgos.

### Validacion

- **D-PPCF (DOS):** AUC de prediccion de pase completado (comparar DOS-aware vs PPCF estandar); correlacion Pearson/Spearman entre valor DOS y xG subsiguiente; calibracion de probabilidades.
- **Modelo causal:** grafico de la curva dosis-respuesta con bandas de confianza; test de placebo (permutacion de angulos); E-values para robustez; comparacion GPS (Hirano-Imbens) vs DML continuo (Colangelo-Lee) como analisis de sensibilidad metodologica; cross-validation temporal (entrenar en primeros N partidos, evaluar en restantes).

---

## BLOQUE 4 — PIPELINE PASO A PASO

El pipeline se divide en 7 fases con dependencias explicitas y tiempos estimados en semanas para un estudiante a tiempo parcial.

**Fase 0 — Infraestructura y datos (semanas 1-2)**
Configurar entorno Python (3.11+), instalar kloppy, mplsoccer, scipy, jenkspy, PyTorch, EconML, DoWhy, causal-curve. Obtener datos de tracking: Metrica Sports open data (2 partidos) para desarrollo, y datos de SkillCorner o StatsBomb 360 si hay acceso institucional. Implementar carga con kloppy y visualizacion con mplsoccer. Dependencias: ninguna. Salida: pipeline de carga de datos funcional.

**Fase 1 — Reimplementacion de D-Def (semanas 2-4)**
Implementar K-Means (k=3) para asignacion de lineas. Computar las 10 variables del vector de estado en cada frame. Calcular deltas a 2, 3 y 4 s. Estandarizar con Z-score. Aplicar PCA y extraer PC1 (longitudinal), PC2 (lateral), PC3 (forma). Calcular D-Def = |PC1| + |PC2| + |PC3|. Validar reproduciendo estadisticas de Goes et al. (2019). Dependencias: Fase 0. Salida: funcion D-Def(pase) -> (D-Def, PC1, PC2, PC3).

**Fase 2 — Deteccion de lineas y features geometricas (semanas 3-5)**
Implementar deteccion de lineas defensivas con jenkspy (3 clusters, ventana 2 s). Computar distancias inter-lineas. Implementar Delaunay intra-equipo con scipy.spatial.Delaunay. Extraer areas de triangulos (Raabe) y **orientacion de aristas** como feature novel. Computar convex hull defensivo. Dependencias: Fase 0. Salida: features geometricas por frame.

**Fase 3 — PPCF base y extension direccional D-PPCF (semanas 4-7)**
Reimplementar PPCF de Spearman usando LaurieOnTracking como referencia, portado a PyTorch (anenglishgoat) para GPU. Parametros base: t_react=0.7s, v_max=5m/s, sigma=0.45s. Definir t_react(theta) = 0.7 + 0.3 x sigmoid((theta - 90)/20) como modulacion direccional (el 0.3 y los parametros de sigmoide se afinaran). Para cada punto (x,y), computar PPCF para N_theta=8 direcciones de llegada del balon. Generar tensor D-PPCF(104, 68, 8). Calcular DOS(x,y) = max_theta - min_theta y DOS_opt(x,y) = argmax_theta. Validar: AUC de prediccion de pase completado comparando PPCF estandar vs D-PPCF. Dependencias: Fase 0. Salida: funcion D-PPCF(frame, theta) y superficie DOS.

**Fase 4 — Feature engineering y confounders (semanas 6-8)**
Computar angulo de pase (arctan2 del vector de desplazamiento del balon). Computar todos los confounders listados en Bloque 3: posicion, velocidad, pressing intensity (unravelsports), compacidad local, PPCF en recepcion, contexto. Ensamblar dataset: cada fila = un pase con T = angulo, Y = D-Def (o PC1/PC2/PC3), X = confounders. Dependencias: Fases 1, 2, 3. Salida: dataset causal completo.

**Fase 5 — Modelo causal DML continuo (semanas 8-11)**
Implementar DML continuo (Colangelo y Lee) usando EconML como base. Primera etapa: entrenar modelos nuisance con XGBoost o LightGBM (gamma_hat(t,x) para E[Y|T=t,X=x] y f_hat(t|x) para GPS). Cross-fitting con L=5 folds. Segunda etapa: estimacion kernel-debiased de beta(t) para t en [0, 360) grados. Graficar curva dosis-respuesta angulo -> D-Def con bandas de confianza al 95%. Repetir para PC1, PC2, PC3 separadamente. Analisis de sensibilidad: E-values, comparacion con GPS clasico (causal-curve), test de placebo. Dependencias: Fase 4. Salida: curva dosis-respuesta y analisis de robustez.

**Fase 6 — Integracion DOS-causal y analisis final (semanas 10-12)**
Correlacionar DOS con xG subsiguiente. Identificar "oportunidades diagonales" como zonas donde DOS es alto y el angulo optimo de DOS coincide con pases de alto D-Def en la curva causal. Generar visualizaciones: mapas de DOS por escenario tactico, curva dosis-respuesta con anotaciones, comparacion PPCF vs D-PPCF. Dependencias: Fases 3, 5. Salida: resultados integrados.

**Fase 7 — Escritura y defensa (semanas 11-14)**
Redaccion del TFG siguiendo estructura estandar. Dependencias: Fase 6.

---

## BLOQUE 5 — TITULOS, NARRATIVA Y PREGUNTA DE INVESTIGACION

### Tres propuestas de titulo

1. **ES:** "Diagonalidad computacional en futbol: estimacion causal de la curva dosis-respuesta angulo-disrupcion y superficies de oportunidad direccional"
   **EN:** "Computational Diagonality in Football: Causal Dose-Response Estimation of Angle-Disruption Effects and Diagonal Opportunity Surfaces"

2. **ES:** "Mas alla del pase hacia adelante: inferencia causal no parametrica del efecto angular sobre la disrupcion defensiva y extension direccional del pitch control"
   **EN:** "Beyond the Forward Pass: Nonparametric Causal Inference of Angular Effects on Defensive Disruption and Directional Pitch Control Extension"

3. **ES:** "El efecto de la direccion: DML continuo para cuantificar como el angulo de accion deshace defensas y pitch control condicionado a la llegada del balon"
   **EN:** "The Direction Effect: Continuous DML for Quantifying How Action Angle Disrupts Defenses and Ball-Arrival-Conditioned Pitch Control"

### Narrativa en 3 frases

La literatura de football analytics ha desarrollado metricas sofisticadas de disrupcion defensiva (D-Def) y control espacial (PPCF) pero ninguna de ellas incorpora la dimension angular de la accion que las genera ni la direccion de llegada del balon al punto de recepcion. Este TFG propone dos contribuciones complementarias: (A) un framework causal basado en Double Machine Learning no parametrico con tratamiento continuo que estima por primera vez la curva dosis-respuesta entre el angulo de un pase y la disrupcion defensiva que produce, y (B) una extension del PPCF denominada Diagonal Opportunity Surfaces que modula la probabilidad de control por la direccion de llegada del balon, revelando zonas donde la eleccion optima de angulo de pase genera la maxima ganancia de control. La integracion de ambos componentes permite cuantificar rigurosamente la "diagonalidad" como concepto tactico emergente, pasando de la intuicion de entrenadores a la evidencia causal.

### Pregunta de investigacion formal

"En que medida el angulo de accion ofensiva (tratamiento continuo, theta en [0, 360)) causa diferencialmente disrupcion defensiva longitudinal, lateral y de forma (medida por los componentes PC1, PC2 y PC3 de D-Def), y como varia la probabilidad de control del terreno de juego (PPCF) en funcion de la direccion de llegada del balon, permitiendo identificar Diagonal Opportunity Surfaces que maximicen simultaneamente control y disrupcion?"

---

## Apendice: Referencias clave con identificadores exactos

| Ref. corta | Autores (anio) | Venue | DOI / arXiv |
|---|---|---|---|
| Goes-2019 | Goes, Kempe, Meerhoff, Lemmink | Big Data 7(1) | 10.1089/big.2018.0067 |
| Forcher-2021 | Forcher, Altmann, Jekauc, Kempe | Entropy 23(12) | 10.3390/e23121607 |
| Forcher-2024 | Forcher et al. | IJSSC 19(2) | 10.1177/17479541231172695 |
| Frencken-2011 | Frencken, Lemmink, Delleman, Visscher | EJSS 11(4) | 10.1080/17461391.2010.499967 |
| Goes-2021 | Goes, Brink, Elferink-Gemser, Kempe, Lemmink | J Sports Sci 39(5) | 10.1080/02640414.2020.1834689 |
| Link-2016 | Link, Lang, Seidenschwarz | PLOS ONE 11(12) | 10.1371/journal.pone.0168768 |
| Spearman-2018 | Spearman | MIT Sloan | (no DOI; ResearchGate 327139841) |
| Fernandez-2018 | Fernandez, Bornn | MIT Sloan | (no DOI; lukebornn.com) |
| Efthimiou-2021 | Efthimiou | arXiv | arXiv: 2107.05714 |
| Efthimiou-2023 | Efthimiou | FICC 2023, Springer | 10.1007/978-3-031-28076-4_37 |
| Bekkers-2025 | Bekkers | arXiv | arXiv: 2501.04712 |
| SoccerMap-2020 | Fernandez, Bornn | ECML-PKDD | 10.1007/978-3-030-67670-4_30 |
| Overmeer-2025 | Overmeer, Janssen, Nuijten | arXiv / icSPORTS | arXiv: 2502.02565; 10.5220/0013784300003988 |
| Cordon-2020 | Cordon-Carmona et al. | IJERPH 17(24) | 10.3390/ijerph17249396 |
| Anzer-2022 | Anzer, Bauer | DMKD 36 | 10.1007/s10618-021-00810-3 |
| Arbues-2020 | Arbues-Sanguesa et al. | CVPR Wkshp | arXiv: 2004.07209 |
| Arbues-2021 | Arbues-Sanguesa et al. | arXiv | arXiv: 2106.00359 |
| Morishita-2025 | Morishita, Aruga, Nakayama, Kijima, Shima | Physica A 666 | 10.1016/j.physa.2025.130507 |
| Herold-2022 | Herold et al. | J Sports Sci 40(12) | 10.1080/02640414.2022.2081405 |
| Cho-2022 | Cho, Ryu, Song | IJSSC 17(2) | 10.1177/17479541211033078 |
| Chernozhukov-2018 | Chernozhukov et al. | Econometrics J 21(1) | 10.1111/ectj.12097 |
| Colangelo-2025 | Colangelo, Lee | JBES | 10.1080/07350015.2025.2505487; arXiv: 2004.03036 |
| Athey-2019 | Athey, Tibshirani, Wager | Ann Stat 47(2) | arXiv: 1610.01271 |
| VanderWeele-2017 | VanderWeele, Ding | Ann Intern Med 167(4) | 10.7326/M16-2607 |
| Alam-2025 | Alam, Moodie, Wu, Swartz | arXiv | arXiv: 2505.11841 |
| Bajons-2023 | Bajons | ACM ICoMS | 10.1145/3613347.3613368 |
| RuizM-2026 | Ruiz-Menarguez, Badiella | arXiv | arXiv: 2602.16830 |
| Ali-2023 | Ali, Yilmaz | IEEE Access 11 | 10.1109/ACCESS.2023.3333878 |
| Narizuka-2019 | Narizuka, Yamazaki | Sci Rep 9 | 10.1038/s41598-019-48623-1 |
| Raabe-2024 | Raabe et al. | J Sports Sci 42(19) | 10.1080/02640414.2024.2414363 |
| Brandes-2025 | Brandes et al. | npj Complexity 2 | 10.1038/s44260-025-00047-x |
| Vater-2024 | Vater | Sci Rep 14 | 10.1038/s41598-024-53706-9 |
| DosSantos-2018 | Dos'Santos et al. | Sports Med 48(10) | 10.1007/s40279-018-0968-3 |
| Mornieux-2014 | Mornieux et al. | J Sports Sci 32(13) | 10.1080/02640414.2013.876508 |
| Ando-2001 | Ando, Kida, Oda | Percept Mot Skills 92(3) | 10.2466/pms.2001.92.3.786 |
| Vater-2019 | Klostermann, Vater, Kredel, Hossner | Front Sports Act Living 1 | 10.3389/fspor.2019.00066 |

---

## Conclusion: tres lagunas, una oportunidad

Este documento confirma tres lagunas investigativas independientes pero convergentes. Primera, la decomposicion direccional de D-Def (PC1/PC2/PC3) nunca ha sido cruzada con el angulo del pase que la genera; la evidencia de Forcher et al. (2021) sobre mayor frecuencia de pases diagonales en ataques exitosos sugiere una relacion causal que nunca se ha formalizado. Segunda, el PPCF de Spearman y todas sus variantes (Fernandez-Bornn, Efthimiou, Bekkers) modelan asimetria por velocidad del jugador pero no por direccion de llegada del balon, ignorando la evidencia biomecanica de que el angulo de excentricidad visual incrementa t_react (Vater, 2024; Ando et al., 2001) y que los COD a angulos mayores requieren mas tiempo (Dos'Santos et al., 2018). Tercera, el DML con tratamiento continuo (Colangelo y Lee, 2025) no se ha aplicado en football analytics; los unicos usos de DML en futbol son con tratamientos categoricos (formaciones en Ruiz-Menarguez, 2026) o binarios (Bajons, 2023). El TFG tiene la oportunidad de ser el primero en conectar estas tres lineas: cuantificar causalmente la diagonalidad como concepto tactico mediante inferencia no parametrica y extender el modelo espacial de referencia del campo con conciencia direccional del balon.