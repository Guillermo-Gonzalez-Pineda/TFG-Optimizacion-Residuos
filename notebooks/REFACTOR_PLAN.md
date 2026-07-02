# Plan de refactorización — módulo `analisis/`

**Proyecto:** TFG-Optimizacion-Residuos (Universidad de La Laguna) — optimización de
localización de contenedores de residuos en San Cristóbal de La Laguna.
**Objeto de este documento:** extraer la lógica duplicada de los cuadernos Jupyter a un
paquete reutilizable `src/python/analisis/`, separando **lógica** (módulos testeables) de
**presentación** (cuadernos narrativos).

> Este fichero es la **única fuente de verdad** del refactor. Tiene dos capas: el **estado
> real** (§0, lo que ya está hecho y verificado) y el **plan + decisiones** (§1–§5, la
> intención y las bifurcaciones resueltas). Se actualiza al cerrar cada fase.

---

## §0 · Estado actual

| Fase | Descripción | Estado | Commit que la cierra |
|------|-------------|--------|----------------------|
| Migración | PICKLE → JSON sin pérdida (escritores + migración) | ✅ hecho | `2a53b3d4`, `a7efbde6` |
| **A** | `rutas.py` (registro de métodos + esquema de rutas) | ✅ hecho | `2366a3b4` |
| **B** | `metricas.py` (métricas puras, geo-free) | ✅ hecho | `2366a3b4`, `d806a58d`, `db14eba8` |
| **C.1** | `estilo.py` + `comparativas.py` (tablas/gráficos, geo-free) | ✅ hecho | `2366a3b4`, `266d0089` |
| **C.2** | `mapas.py` (visualización geográfica, osmnx/geopandas) | ✅ hecho | `82bba2ab` |
| (extra) | `geo.py` (número + geo: densidad, validación de grafo) | ✅ hecho | `ad75be3e` |
| **D** | Refactor de cuadernos (ver roadmap abajo) | ⏳ en curso | 01 ✅ `8c5aa4b4` |
| **E** | Migración física de rutas + limpieza de cruft | ⏳ pendiente | — |

### Roadmap de cuadernos (Fase D) — 6 cuadernos

| # | Cuaderno | Rol | Estado / orden |
|---|----------|-----|----------------|
| 01 | Validación de instancias | validar | ✅ hecho (`8c5aa4b4`) |
| 00 | Generación (pipeline OSMnx) | producir datos | ⏸️ **aplazado** (decisión del autor) |
| 02 | Modelo exacto | 1 método en profundidad | **siguiente** |
| 03 | Relajación lagrangiana | 1 método (loader roto → se arregla en `cargar_solucion`) | tras 02 |
| 04 | **Metaheurística (tabú)** | 1 método (consume el JSON del C++) | tras 03 |
| 05 | **Comparaciones** | **capstone: cruza los 3 métodos** | **el último** |

**Orden efectivo:** `01 ✅ → 02 → 03 → 04 (metaheurística) → 05 (comparaciones)`. El **00 está
aparcado** por decisión del autor (no es el siguiente). El de **comparaciones va forzosamente al
final**: necesita que los loaders de los tres métodos funcionen y que sus artefactos existan.

**Siguiente paso:** cuaderno **02** (modelo exacto), formato de solución limpio. Requiere
inventario de lectura previo, como se hizo con el 01. El 00 (cuando se retome) tiene celdas
diagnóstico rotas (c28–30) y un `instancia_laguna.json` sin sufijo de tamaño (c25).

**Ventaja acumulada:** el capstone de comparaciones ya está medio construido —
`comparativas.comparar_metodos(sols_por_metodo, referencia="exacto")` y `tabla_resumen(sols)` se
diseñaron **agnósticas al método** pensando en él. La metaheurística ya tiene al menos un
artefacto en disco (`output/metaheuristica/solucion_tabu_1500m.json`).

**Bloqueo conocido:** el cuaderno **03** no ejecuta contra los artefactos actuales (lee
`best_feasible` / `runtime_seconds`, que ya no existen). Se arregla solo al pasar por
`cargar_solucion`. Por eso el 03 no adopta la política de outputs hasta *después* de
refactorizarse.

### Módulos del paquete `analisis/` (todos commiteados)

| Módulo | Cuadrante | Contenido | Depende de |
|--------|-----------|-----------|------------|
| `serializacion.py` | — | (De)serialización anidada `{a:{b:v}}`, coerción numpy→nativo | ninguna pesada |
| `carga.py` | — | `cargar_solucion` → normaliza el formato en dicts canónicos | serializacion |
| `rutas.py` | — | `Metodo`, `REGISTRO`, `raiz_repo()`, constructores de ruta | `pathlib` |
| `metricas.py` | número · geo-free | métricas de solución e instancia | numpy |
| `comparativas.py` | visual · geo-free | tablas y gráficos comparativos | pandas, matplotlib |
| `geo.py` | número · geo | densidad convex-hull, validación de grafo | shapely, pyproj, networkx, osmnx |
| `mapas.py` | visual · geo | toda la visualización geográfica | osmnx, geopandas, shapely |

**Invariante del paquete (verificado 3 veces en runtime):** `import analisis` es **geo-free**
— `'geopandas'`/`'osmnx'` NO aparecen en `sys.modules` tras importarlo. `geo` y `mapas` solo
se cargan bajo demanda (`from analisis import geo`). `__init__.py` NO los re-exporta de forma
ansiosa.

---

## §1 · Objetivo y arquitectura

### Principio de diseño

El **método** (exacto / lagrangiana / greedy / metaheurística) entra solo como un `str`;
`carga.py` devuelve un objeto/estructura de solución **canónica**, y a partir de ahí
`metricas` / `mapas` / `comparativas` **no saben de qué método vienen**. La unificación entre
métodos vive en la **capa de carga**, no en renombrar campos.

### La matriz de análisis (2×2)

La lógica de análisis se organiza según dos ejes: si el cómputo produce **números o dibujos**,
y si necesita **la pila geoespacial o no**.

|              | **geo-free**        | **geo-dependiente** |
|--------------|---------------------|---------------------|
| **número**   | `metricas.py`       | `geo.py`            |
| **visual**   | `comparativas.py`   | `mapas.py`          |

Esta separación es la que garantiza el invariante geo-free: `carga` y `metricas` corren en un
entorno sin cartografía (CI, o cuadernos que solo necesitan cifras), sin arrastrar `osmnx`.

---

## §2 · Fases

### Fase A — `rutas.py` (✅)

Registro de métodos como única fuente de verdad y esquema de rutas anclado al repositorio.

- `@dataclass(frozen=True) Metodo(clave, adjetivo)` — `clave` = carpeta, `adjetivo` = fichero.
  Resuelve los desajustes de nombre: `exacto`→`exacta`, `metaheuristica`→`tabu`.
- `REGISTRO: dict[str, Metodo]` — `exacto`, `lagrangiana`, `greedy`, `metaheuristica`
  (los dos últimos, sin artefactos aún, incluidos como dato puro para no crear casos especiales).
- `raiz_repo()` — sube hasta encontrar `.git`. **Mata la dependencia del `cwd`** (causa de los
  árboles fantasma `src/python/output/` y `output/relajacion_{size}m/`).
- Constructores del **esquema nuevo**: `ruta_solucion_json`, `ruta_instancia`, `ruta_grafo`,
  `ruta_buildings`, `ruta_figura`, `tamaños_disponibles`.
- **No importa** `gurobipy`, `osmnx` ni `instancia`. Puro `pathlib`.

> Describe el esquema **nuevo** (`output/<metodo>/solucion_<adjetivo>_<tam>m.json`); los
> ficheros físicos siguen en el layout **viejo** hasta la Fase E. Por eso
> `tamaños_disponibles("exacto")` devuelve `[]` hoy — es correcto, no un bug (ver decisión D6).

### Fase B — `metricas.py` (✅)

Funciones **puras**, sin matplotlib ni osmnx, agnósticas al método.

*De solución:* `puntos_abiertos`, `n_puntos_abiertos`, `total_bins`, `bins_por_tipo`, `coste`,
`gap` (usa `gap` en lagrangiana o `gap_gurobi` en exacto), `desglose_coste`,
`violaciones_capacidad`, `resumen`, `demanda_por_punto`.

*De instancia:* `outliers_demanda_iqr`, `cobertura_por_tipo`,
`candidatos_reales_vs_artificiales`, `resumen_distancias`.

**Validado:** 15/15 instancias exactas contra su resumen; anclas del enunciado
(1500m → `2265500.0` / 403 puntos; 500m → `343070.0` / 64 puntos). Ver invariantes en §4.

### Fase C — presentación (✅)

- **C.1 — `estilo.py` + `comparativas.py`** (matplotlib + pandas, sin osmnx).
  - `estilo.py`: constantes puras — `TIPOS_RESIDUO`, `COLOR_*`, `CMAP_*`, `PALETA_METODOS`
    (claves = claves del `REGISTRO`), `nombre_tipo`.
  - `comparativas.py`: `tabla_resumen`, `tabla_matplotlib`, `grafico_escalabilidad`,
    `comparar_metodos`, `grafico_convergencia`, `tabla_instancias`.
- **C.2 — `mapas.py`** (osmnx / geopandas / shapely; toda la dependencia pesada aislada).
  - `cargar_fondo`, `dibujar_calles`, `dibujar_edificios`, `emparejar_demanda`,
    `mapa_solucion` (función **única** que sustituye a `plot_exact_solution` +
    `plot_rich_solution` + `plot_rich_solution_lagrangian`), `mapa_instancia`, `mapa_demanda`.
  - GeoJSON cargado **sin fiona** (`json.load` + `shapely.shape`), porque geopandas 1.x usa
    `pyogrio` y `fiona` no está instalado.
  - `emparejar_demanda`: empareja por centroide con umbral `< 0.0005°`. El aviso de `.centroid`
    sobre EPSG:4326 es un **falso positivo** aquí (solo se mide cercanía, no áreas); se suprime
    ese aviso concreto y **no** se reproyecta (reproyectar cambiaría el umbral).

### (extra) `geo.py` — el cuarto cuadrante (✅)

Cómputos que **necesitan** la pila geoespacial pero devuelven **números**, no dibujos. No
pueden vivir en `metricas` (rompería el invariante geo-free) ni en `mapas` (no son dibujo).

- `densidad_convexhull(inst)` → `{area_km2, densidad_hab_km2}`. Reproyecta a UTM
  (`estimate_utm_crs`) porque **mide superficie** — caso opuesto a `emparejar_demanda`, donde no
  se reproyecta. Usa `union_all()` (API vigente, no el deprecado `unary_union`).
- `validar_grafo(inst_o_tam)` → `{n_nodos, n_aristas, n_componentes_conexas, es_conexo,
  n_nodos_artificiales, n_aristas_largas, n_nodos_aislados}`.

### Fase D — refactor de cuadernos (⏳ en curso)

Orden efectivo por acoplamiento creciente: **01 ✅ → 02 → 03 → 04 (metaheurística) →
05 (comparaciones)**; el **00 está aparcado**. Cada cuaderno pasa de lógica inline a
**narrativa + llamadas** a `analisis.*`. Se ejecuta de arriba abajo en kernel limpio y se
comparan los números clave contra los validados (no debe haber regresión). Todos aplican la
política de outputs (D7) y de muestra (D9).

- **01 (✅)** — validación de instancias. De 495 → **159 líneas de código**; se eliminaron las
  3 funciones inline `validar_grafo` / `validar_demanda` / `validar_distancias_cobertura`.
  Radios de cómputo `[500, 1000, 1500]`; mapas solo `[500, 1500]`. Taxonomía corregida a la
  canónica, rutas literales sustituidas por `rutas.*`. 11/11 números clave sin regresión.
- **00** — pipeline de generación (OSMnx → build → save). Arreglar de paso las celdas
  diagnóstico rotas (c28–30) y el `instancia_laguna.json` sin sufijo (c25).
- **02** — modelo exacto. Migrar la carga a `cargar_solucion`; usar `mapa_solucion`.
- **03** — relajación lagrangiana. Al pasar por `cargar_solucion` se arregla el loader roto
  (`best_feasible` / `runtime_seconds` / `y_assign` como array). Poblar outputs **solo** tras
  arreglarlo. Muestra real ≈ solo 500m hasta correr el barrido (ver D9).
- **04 — metaheurística (tabú).** Consume el JSON producido por el motor C++
  (`output/metaheuristica/solucion_tabu_<tam>m.json`). Mismo esqueleto que 02/03; `mapa_solucion`
  sirve sin cambios (la solución llega ya normalizada por `cargar_solucion`).
- **05 — comparaciones (capstone).** Cruza los tres métodos con
  `comparativas.comparar_metodos(..., referencia="exacto")` y `tabla_resumen`. Va el último:
  requiere loaders y artefactos de los tres. Casi solo narrativa + llamadas.

### Fase E — migración física de rutas + limpieza (⏳ pendiente, la última)

Va al final porque los **lectores** (cuadernos, ya sobre `rutas.py`) deben tolerar el layout
nuevo **antes** de tocar los escritores. Incluye migrar los writers al esquema nuevo, un script
one-off para mover/renombrar artefactos, borrar los `.pkl` (red de seguridad hasta aquí) y
eliminar el cruft. Ver mapeo en §5.

---

## §3 · Decisiones tomadas

> Registro de las bifurcaciones resueltas durante el refactor. Es la sección más valiosa: aquí
> viven las decisiones que de otro modo se perderían al cambiar de dispositivo.

- **D1 · Migración PICKLE → JSON, sin pérdida.** La (de)serialización es **única y compartida**
  por escritores y migración, así que lo reescrito hoy es bit-idéntico a lo que generarán las
  ejecuciones futuras. Los `numpy.float64` (`best_lb`, `gap`, `lb_history`) se coaccionan a
  `float` nativo: mismo doble IEEE-754, round-trip de igualdad **exacta**. Verificado sobre 16
  artefactos, 0 diferencias.

- **D2 · Nombres de campo en disco = los del pickle (`z`/`x`/`w`/`y_assign`), NO alineados con
  C++.** El formato en disco tiene campos que C++ no tiene (`w`, `gap_gurobi`, `status`,
  `lb_history`); renombrar el subconjunto coincidente daría un esquema mitad-pickle mitad-C++.
  La convergencia entre métodos se hace en la **capa de carga** (objeto canónico), no en disco.
  Estructuras canónicas: `z {j:int}`, `x`/`w {(j,k):int}`, `y_assign {(i,k):j}`.

- **D3 · `desglose_coste` = hipótesis H2 (apertura por candidato).** La parte fija se suma
  **punto a punto** con `inst.J[j].opening_cost`, que **varía por candidato** (default 4000,
  pero p. ej. `J[0]=3800`). El escalar `params.opening_cost=4000` es solo la base y **no** es lo
  que usa la métrica. La identidad `fija + variable == coste` **no es tautológica** (se calculan
  de forma independiente): valida que el objetivo del modelo exacto es exactamente
  *apertura + bins*, sin que `overflow_penalty` ni NIMBY aporten al óptimo determinista (HDM).
  ⚠️ Para métodos cuyo objetivo incluya penalizaciones (greedy/tabú), esa identidad podría no
  cumplirse: el docstring es específico del exacto.

- **D4 · Taxonomía canónica de residuos** (`estilo.TIPOS_RESIDUO`, única fuente):
  `{0: "Orgánica", 1: "Resto", 2: "Reciclable", 3: "Peligrosos"}`. La antigua
  "Orgánico / Resto / Papel-Cartón / Vidrio" era **deriva de copy-paste**, no el canon del
  modelo. Los radios de cobertura lo confirman: `{0:125, 1:125, 2:175, 3:225}` — `k=3` con radio
  máximo es *peligrosos* (se genera poco, el vecino camina más), no vidrio.

- **D5 · Arquitectura 2×2 + `geo.py` como cuarto cuadrante.** Los cómputos geo que devuelven
  números (densidad, validación de grafo) van a `geo.py`, no a `metricas` (que debe seguir
  geo-free) ni a `mapas` (que es dibujo). Invariante `import analisis` geo-free, verificado.

- **D6 · `rutas.py` como spec puro (Opción 1, sin *shim* de compatibilidad).** Describe solo el
  esquema nuevo; `tamaños_disponibles` devuelve `[]` hasta la Fase E. Durante D, los
  verificadores y cuadernos cargan por **ruta literal del layout viejo**
  (`output/exacto_<tam>m/solucion_exacta.json`).

- **D7 · Política de outputs en cuadernos.** Se versionan **con** outputs ejecutados
  (autocontenidos para el tribunal); se limpia solo **metadata volátil** (`execution_count`,
  timestamps de ejecución) sin tocar los outputs visibles; se ejecutan con **kernel limpio**
  antes de commitear. **NO** se usa `nbstripout` (borraría outputs). Cada cuaderno adopta la
  política **al refactorizarse**, no antes; el 03, solo tras arreglarse.

- **D8 · Orientación de `dij` y marcador de candidato artificial (hallazgos verificados).**
  `dij` está orientado como `dij[j][i]` (clave externa = candidato `j`, interna = edificio `i`).
  Candidato **artificial** = `osm_id` que empieza por `"-"` (no hay booleano ni prefijo `999`).
  Nota: los nodos artificiales del **grafo** guardado son 0 (los candidatos artificiales viven
  en el conjunto de candidatos de la instancia, no en el graphml).

- **D9 · Política de muestra en cuadernos (tres niveles).** Los cuadernos por método NO imprimen
  las 15 instancias. Se distingue según qué se multiplica con el número de tamaños:
  - **Tablas** → **todas** las instancias (barato; se quiere ver el barrido completo).
  - **Gráficos agregados** (escalabilidad, gap-vs-tamaño: *un* gráfico que resume el rango) →
    **todos** los tamaños (la tendencia los necesita; muestrear escondería la curva).
  - **Visuales por instancia** (mapas de solución, convergencia LB/UB individual) → **muestra**
    `RADIOS_MUESTRA = [500, 800, 1000, 1500]` (mínimo legible + interior + máximo de memoria).

  `RADIOS_MUESTRA` es **única fuente de verdad**, vive en `estilo.py`. La muestra real de cada
  cuaderno = `RADIOS_MUESTRA ∩ artefactos disponibles` (la lagrangiana hoy solo tiene 500m
  resuelto, así que su cuaderno mostrará una instancia hasta que se corra el barrido). Beneficio
  colateral: mantiene el peso del `.ipynb` con outputs (D7) en el orden de ~2 MB, no ~10 MB.

- **D10 · Estado de optimalidad de Gurobi (`estilo.ESTADO_GUROBI` / `estado_gurobi`).** La
  calidad de una solución exacta la decide el **`status` CRUDO** de Gurobi, **no el gap
  redondeado**: un `gap ≈ 0` puede venir de una solución detenida, no demostrada óptima. Mapeo
  canónico `{2: "óptimo", 9: "time-limit", 11: "interrumpido"}`; `estado_gurobi(status)` devuelve
  la etiqueta o el código crudo (`"status <n>"`) si no está en el mapa — **nunca asume óptimo**
  (p. ej. `13`=SUBOPTIMAL → `"status 13"`). Lectura semántica: **9 = censura por el límite de
  tiempo** (la meseta de 4 h) frente a **11 = interrupción externa antes del límite** (dos cosas
  distintas). Verificado por diagnóstico de solo lectura: status `2` en 250–650 m, status `9` en
  700–1000 m (clavados en 14400 s), status `11` solo en 1500 m (8178 s < 14400 s → interrupción
  real, no time-limit). El mapeo **y** el marcador visual `MARCADOR_ESTADO = {"time-limit": "s",
  "interrumpido": "^"}` viven en `estilo.py` como **única fuente**; el módulo **agnóstico**
  `comparativas` NO los conoce: `grafico_escalabilidad(resaltar=[grupos])` recibe del cuaderno la
  lista de grupos `{indices, etiqueta, marcador}` y solo dibuja los índices dados (la
  clasificación por `status` la hace el cuaderno del exacto, no la librería). ⚠️ El 1500 m es
  **dato volátil** (re-ejecución pendiente de `git pull`): el cuaderno lo lee dinámicamente de la
  solución, no lo clava.

---

## §4 · Riesgos y validaciones

### Invariantes usados para validar (el sello de calidad del refactor)

- **Conservación de demanda:** `Σ_j demanda_por_punto(sol, inst, k)[j] == total_population`,
  idéntico para todo `k`. Comprobado: 4 tipos coinciden bit a bit (39802.005 en 1500m). Prueba
  la correcta inversión de `y_assign`.
- **Monotonía de cobertura:** `edificios_sin_cobertura` no crece al aumentar el radio
  (`acc_max` 19→19→30→41 en 1500m). Valida de paso la orientación de `dij` (si estuviera
  invertida, la monotonía se rompe).
- **Partición de conjuntos:** `reales + artificiales == n_candidates`;
  `keys(demanda) ⊆ puntos_abiertos`; `bins_por_tipo` suma a `total_bins`.
- **Conservación de conexiones:** `resumen_distancias.n == inst.n_dijkstra_connections`.
- **Desglose de coste:** `fija + variable == coste` (1500m: 1.723.400 + 542.100 = 2.265.500).
- **Validación EXTERNA (no auto-consistencia):** densidad por convex-hull ≈ INE 4471 hab/km²
  en los tres tamaños (500m→4513, 1000m→5053, 1500m→4446; error < 1% en 1500m). La instancia
  sintética reproduce la densidad oficial de La Laguna.
- **Aislamiento geo:** `import analisis` no arrastra `osmnx`/`geopandas` (`sys.modules`).

### Hallazgos para la memoria (no son bugs)

- **Asimetría de la demanda:** los outliers IQR acumulan una fracción **grande y creciente** de
  la demanda (500m→0.565, 1000m→0.662, 1500m→0.666). Enmarcar como *"demanda urbana fuertemente
  asimétrica, creciente con la escala"*, no como *"edificios anómalos"*. El IQR asume cierta
  simetría; con `h_i` muy sesgado marca como atípica la estructura densa normal de la ciudad.
- **Aristas largas (grafo 1500m: 26 tramos > 500 m):** justifican **cuantitativamente** la
  decisión de insertar candidatos artificiales para densificar candidatos en tramos largos.
- **Lagrangiana 500m alcanza el óptimo exacto** (`gap_vs_ref = 0`, mismo coste 343.070). Matizar
  en la defensa si es la *misma* solución o *una* de igual coste (comparar puntos/bins).

### Riesgos originales (y cómo se neutralizaron)

| Riesgo | Neutralización |
|--------|----------------|
| Deriva de formato / acoplamiento al pickle | Carga normalizadora (`cargar_solucion`) |
| Estado compartido / orden de celdas | Funciones puras; cuadernos = narrativa + llamadas |
| Dependencia del `cwd` | `rutas.raiz_repo()` ancla a `.git` |
| Gurobi | `analisis` no importa `gurobipy` (ni transitivamente) |
| OSMnx/geopandas/fiona | Aislados en `geo.py` y `mapas.py` |
| Orientación de `dij` | Fijada `dij[j][i]`, testeada por monotonía de cobertura |
| Mapeo adjetivo del nombre | `REGISTRO` de `rutas.py` como única fuente |

---

## §5 · Mapeo de rutas viejo → nuevo (para la Fase E)

**Estado actual:** la migración PICKLE→JSON cambió la **extensión** pero conservó el layout
**viejo**. Hoy las soluciones viven en `output/<metodo>_<tam>m/solucion_<adjetivo>.json`.

| Actual (layout viejo, ya en JSON) | Nuevo (Fase E) | Acción |
|-----------------------------------|----------------|--------|
| `output/exacto_1500m/solucion_exacta.json` | `output/exacto/solucion_exacta_1500m.json` | mover/renombrar |
| `output/exacto_1500m/solucion_exacta_resumen.json` | `output/exacto/solucion_exacta_1500m_resumen.json` | mover/renombrar |
| `output/lagrangiana_500m/solucion_lagrangiana.json` | `output/lagrangiana/solucion_lagrangiana_500m.json` | mover/renombrar |
| `output/barrido_relajacion/barrido_*` | `output/lagrangiana/barrido_*` | mover |
| `output/*.png` (tablas/convergencia/escalabilidad) | `output/figuras/*.png` | mover |
| `output/<metodo>_run.log` | `output/<metodo>/logs/` | mover |
| `data/processed/instancia_laguna.json` (00 c25, sin sufijo) | `data/processed/instancia_laguna_500m.json` | renombrar (convención) |

**A eliminar (cruft):**

- `output/**/*.pkl` — red de seguridad **hasta la Fase E**; borrar solo entonces (incluye el
  legacy `output/solucion_exacta.pkl` de raíz y los duplicados de `src/python/output/`).
- `output/relajacion_{size}m/` (literal, `{size}` sin expandir) — cruft de una versión vieja del
  script.

**Escritores a migrar** (vía `rutas.py`): `modelo_exacto.py`, `lagrangiana/persistencia.py`,
`scripts/barrido_instancias.py`.

**Pendiente de clasificar:** `output/greedy/` (untracked) — confirmar si es resultado canónico o
basura de pruebas.