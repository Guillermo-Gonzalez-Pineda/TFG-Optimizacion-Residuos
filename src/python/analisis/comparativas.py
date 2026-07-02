"""
Tablas y gráficos comparativos entre métodos (exacto, lagrangiana, greedy, tabú).

Consume el formato común de solución (el dict que devuelve
``analisis.carga.cargar_solucion``) a través de ``analisis.metricas``, sin
reimplementar ninguna métrica. Permitido aquí: pandas y matplotlib. PROHIBIDO
(ni directa ni transitivamente): osmnx, geopandas, fiona, gurobipy.

NO importa ``analisis.carga``: la carga de ficheros es responsabilidad de quien
llama (verificador/cuadernos). Aquí solo se reciben soluciones YA cargadas, lo
que mantiene este módulo desacoplado del disco y del esquema de rutas.

Sobre el contenedor ``Solucion``: ``cargar_solucion`` devuelve un dict SIN método
ni tamaño (descarta la clave ``metodo`` y nunca lleva ``tam``). Como las tablas
comparativas necesitan etiquetar cada fila por método y tamaño, este contenedor
los aporta desde el sitio de carga (que sí conoce la ruta de la que vino la
solución). No duplica datos de la solución: solo la envuelve con sus etiquetas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from . import estilo
from . import metricas


@dataclass
class Solucion:
    """Solución cargada + sus etiquetas de contexto.

    - ``metodo``: clave de método (coincide con rutas.REGISTRO: "exacto", ...).
    - ``tam``: tamaño de la instancia en metros.
    - ``datos``: el dict que devuelve ``analisis.carga.cargar_solucion``.
    - ``inst``: ``Instance`` opcional (necesaria solo para desglose/violaciones).
    """

    metodo: str
    tam: int
    datos: dict
    inst: Any = None


def _datos(sol: "Solucion | dict") -> dict:
    """Acepta tanto el contenedor ``Solucion`` como el dict crudo de
    ``cargar_solucion``, y devuelve el dict de datos."""
    return sol.datos if isinstance(sol, Solucion) else sol


def _fmt(v: Any) -> str:
    """Formatea un valor para una celda de tabla (legible, agnóstico a columna)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, bool):
        return "Sí" if v else "No"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v.is_integer():
            return f"{int(v):,}"
        return f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4g}"
    return str(v)


def tabla_resumen(sols: Sequence[Solucion]) -> pd.DataFrame:
    """Una fila por solución (cualquier método), vía ``metricas.resumen``.

    Columnas: metodo, tam, coste, n_puntos, n_bins, gap, tiempo. Agnóstica al
    método (``gap`` funciona para exacto y lagrangiana). ``tiempo`` se lee del
    campo crudo ``runtime`` (no es una métrica, es un escalar persistido)."""
    filas = []
    for s in sols:
        m = metricas.resumen(s.datos, s.inst)
        filas.append({
            "metodo":   s.metodo,
            "tam":      s.tam,
            "coste":    m["coste"],
            "n_puntos": m["n_puntos_abiertos"],
            "n_bins":   m["total_bins"],
            "gap":      m["gap"],
            "tiempo":   s.datos.get("runtime"),
        })
    df = pd.DataFrame(filas, columns=["metodo", "tam", "coste", "n_puntos",
                                      "n_bins", "gap", "tiempo"])
    return df.sort_values(["metodo", "tam"]).reset_index(drop=True)


# Columnas (en español) de la tabla comparativa de instancias, en orden.
_COLS_INSTANCIAS = [
    "tam", "n_edificios", "n_candidatos", "poblacion", "reales", "artificiales",
    "sin_cobertura_algun_tipo", "n_outliers", "pct_demanda_outliers",
    "dist_media", "dist_max", "area_km2", "densidad_hab_km2",
    "grafo_conexo", "aristas_largas",
]


def _sub(fila: dict, bloque: str, clave: str) -> Any:
    """Lee ``fila[bloque][clave]`` con tolerancia: NaN si falta el bloque o la
    clave (así una instancia sin algún cómputo no rompe la tabla)."""
    sub = fila.get(bloque)
    return sub.get(clave, math.nan) if isinstance(sub, dict) else math.nan


def tabla_instancias(filas: Sequence[dict]) -> pd.DataFrame:
    """Una fila por instancia, aplanando dicts de métricas/geo YA calculados.

    ``comparativas`` es geo-free: esta función NO calcula métricas ni importa
    ``geo``/``mapas``; solo FORMATEA lo que recibe (igual que ``tabla_resumen``
    recibe objetos ``Solucion``). Quien llame — el cuaderno — invoca ``metricas.*``
    y ``geo.*`` y pasa aquí, por instancia, un dict con estos bloques:

        {"tam", "n_buildings", "n_candidates", "total_population",
         "cobertura":  metricas.cobertura_por_tipo(inst),
         "outliers":   metricas.outliers_demanda_iqr(inst),
         "candidatos": metricas.candidatos_reales_vs_artificiales(inst),
         "distancias": metricas.resumen_distancias(inst),
         "densidad":   geo.densidad_convexhull(inst),
         "grafo":      geo.validar_grafo(inst)}

    Cualquier bloque o clave ausente se rellena con NaN (no rompe la fila)."""
    registros = []
    for fila in filas:
        registros.append({
            "tam":                      fila.get("tam", math.nan),
            "n_edificios":              fila.get("n_buildings", math.nan),
            "n_candidatos":             fila.get("n_candidates", math.nan),
            "poblacion":                fila.get("total_population", math.nan),
            "reales":                   _sub(fila, "candidatos", "reales"),
            "artificiales":             _sub(fila, "candidatos", "artificiales"),
            "sin_cobertura_algun_tipo": _sub(fila, "cobertura", "sin_cobertura_algun_tipo"),
            "n_outliers":               _sub(fila, "outliers", "n_outliers"),
            "pct_demanda_outliers":     _sub(fila, "outliers", "pct_demanda_outliers"),
            "dist_media":               _sub(fila, "distancias", "media"),
            "dist_max":                 _sub(fila, "distancias", "max"),
            "area_km2":                 _sub(fila, "densidad", "area_km2"),
            "densidad_hab_km2":         _sub(fila, "densidad", "densidad_hab_km2"),
            "grafo_conexo":             _sub(fila, "grafo", "es_conexo"),
            "aristas_largas":           _sub(fila, "grafo", "n_aristas_largas"),
        })
    df = pd.DataFrame(registros, columns=_COLS_INSTANCIAS)
    return df.sort_values("tam").reset_index(drop=True)


def tabla_matplotlib(df: pd.DataFrame, titulo: str | None = None, ax=None):
    """Renderiza un DataFrame como tabla estilada (cabecera oscura con texto
    blanco en negrita, filas alternas). Unifica el estilo de 02c3 y 03c5."""
    columnas = list(df.columns)
    if ax is None:
        alto = max(2.0, 0.45 * len(df) + 1.0)
        ancho = max(8.0, 1.7 * len(columnas))
        _, ax = plt.subplots(figsize=(ancho, alto))
    ax.axis("off")

    celdas = [[_fmt(v) for v in fila] for fila in df.itertuples(index=False)]
    tabla = ax.table(cellText=celdas, colLabels=columnas,
                     loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.5)

    # Cabecera
    for j in range(len(columnas)):
        celda = tabla[0, j]
        celda.set_facecolor(estilo.COLOR_CABECERA)
        celda.set_text_props(color=estilo.COLOR_TEXTO_CABECERA, fontweight="bold")

    # Filas alternas
    for i in range(len(celdas)):
        color = estilo.COLOR_FILA_PAR if i % 2 == 0 else estilo.COLOR_FILA_IMPAR
        for j in range(len(columnas)):
            tabla[i + 1, j].set_facecolor(color)

    if titulo:
        ax.set_title(titulo, fontsize=14, fontweight="bold", pad=12)
    return ax


def grafico_escalabilidad(sols: Sequence[Solucion],
                          columnas: Sequence[str] = ("coste", "tiempo", "n_puntos"),
                          ax=None,
                          resaltar: "list[dict] | None" = None):
    """Un panel por columna (eje X = tam), una línea por método. Versión genérica
    de 02c8. Devuelve el/los ejes.

    Columnas dibujables: cualquiera de ``tabla_resumen`` (``coste``, ``tiempo``,
    ``gap``, ``n_puntos``, ``n_bins``). El panel de ``tiempo`` usa escala LOGARÍTMICA
    (crece varios órdenes de magnitud); el de ``gap`` usa escala LINEAL con el eje Y en
    porcentaje.

    ``resaltar`` (opcional): LISTA de grupos, cada uno
    ``{"indices": [...], "etiqueta": str, "marcador": str}``. Cada grupo se dibuja en el
    panel de ``tiempo`` con su ``marcador`` sobre los puntos indicados —por su POSICIÓN
    en ``sols``— y una entrada de leyenda con su ``etiqueta``. Es AGNÓSTICO: esta función
    solo dibuja los grupos que recibe; NO importa ``estado_gurobi``, NO sabe de
    ``status`` ni de Gurobi. Motivación (en el exacto): el tiempo se satura en el límite
    de 4 h, así que esa meseta es CENSURA (time-limit), no convergencia; marcarla evita
    leerla como estabilización, y el panel de ``gap`` revela el error real que hay
    detrás. La clasificación de cada grupo (time-limit vs interrumpido) la hace el
    CUADERNO (Tarea B, vía ``estilo.estado_gurobi``/``MARCADOR_ESTADO``), no la librería.

    RETROCOMPATIBLE: una llamada SIN ``gap`` en ``columnas`` y SIN ``resaltar`` produce
    exactamente el gráfico de 3 paneles anterior (lo llaman así 03/04/05)."""
    df = tabla_resumen(sols)

    if ax is None:
        _, ax = plt.subplots(1, len(columnas),
                             figsize=(6 * len(columnas), 5), squeeze=False)
        ejes = ax[0]
    else:
        # Permite pasar un único Axes o un array de ejes.
        ejes = list(ax) if hasattr(ax, "__len__") else [ax]

    etiquetas = {"coste": "Coste (€)", "tiempo": "Tiempo (s)",
                 "n_puntos": "Puntos abiertos", "n_bins": "Contenedores",
                 "gap": "Gap (%)"}

    for eje, col in zip(ejes, columnas):
        for metodo, grupo in df.groupby("metodo"):
            g = grupo.dropna(subset=[col]).sort_values("tam")
            if g.empty:
                continue
            eje.plot(g["tam"], g[col], "o-", linewidth=2, markersize=6,
                     color=estilo.PALETA_METODOS.get(metodo), label=metodo)
        eje.set_xlabel("Tamaño de instancia (m)")
        eje.set_ylabel(etiquetas.get(col, col))
        eje.set_title(f"{etiquetas.get(col, col)} vs tamaño")
        eje.grid(True, alpha=0.3)
        if col == "tiempo":
            eje.set_yscale("log")     # el tiempo crece varios órdenes de magnitud
        if col == "gap":
            # Escala LINEAL (no log) y eje Y en %: el gap es la métrica que revela el
            # error real detrás de la meseta del time-limit.
            eje.yaxis.set_major_formatter(PercentFormatter(xmax=1))

        # Leyenda: por defecto solo si hay varios métodos (comportamiento anterior);
        # el resaltado del panel de tiempo la fuerza para explicar los marcadores.
        mostrar_leyenda = df["metodo"].nunique() > 1
        if resaltar and col == "tiempo":
            for grupo in resaltar:
                indices = grupo.get("indices", [])
                etiqueta = grupo.get("etiqueta", "resaltado")
                marcador = grupo.get("marcador", "s")
                xs = [sols[i].tam for i in indices]
                ys = [sols[i].datos.get("runtime") for i in indices]
                if xs:
                    eje.scatter(xs, ys, marker=marcador, s=140, facecolors="none",
                                edgecolors="#c0392b", linewidths=1.8, zorder=6,
                                label=etiqueta)
                    mostrar_leyenda = True
        if mostrar_leyenda:
            eje.legend()

    return ejes


def comparar_metodos(sols_por_metodo: dict[str, list],
                     referencia: str = "exacto") -> pd.DataFrame:
    """Alinea por tamaño y mide la distancia al método de referencia.

    Para cada método != referencia y cada tamaño común:
        gap_vs_ref = (coste_metodo - coste_ref) / coste_ref

    Generaliza 03c9 (que comparaba el UB lagrangiano con el óptimo de Gurobi).
    Devuelve un DataFrame: tam, metodo, coste, coste_ref, referencia, gap_vs_ref.
    """
    if referencia not in sols_por_metodo:
        raise KeyError(
            f"Método de referencia {referencia!r} no está en "
            f"{sorted(sols_por_metodo)}."
        )

    coste_ref = {s.tam: metricas.coste(s.datos) for s in sols_por_metodo[referencia]}

    filas = []
    for metodo, lista in sols_por_metodo.items():
        if metodo == referencia:
            continue
        for s in lista:
            c = metricas.coste(s.datos)
            cref = coste_ref.get(s.tam)
            gap_vs_ref = None
            if c is not None and cref not in (None, 0):
                gap_vs_ref = (c - cref) / cref
            filas.append({
                "tam":        s.tam,
                "metodo":     metodo,
                "coste":      c,
                "coste_ref":  cref,
                "referencia": referencia,
                "gap_vs_ref": gap_vs_ref,
            })
    return pd.DataFrame(
        filas,
        columns=["tam", "metodo", "coste", "coste_ref", "referencia", "gap_vs_ref"],
    ).sort_values(["metodo", "tam"]).reset_index(drop=True)


def grafico_convergencia(sol_lagrangiana: "Solucion | dict", ax=None):
    """LB/UB vs iteración a partir de ``lb_history``/``ub_history``. Específico de
    la lagrangiana. Si esas claves no están presentes (p. ej. una solución
    exacta), no inventa nada: dibuja un aviso claro y devuelve el eje."""
    datos = _datos(sol_lagrangiana)
    lb = datos.get("lb_history")
    ub = datos.get("ub_history")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    if not lb or not ub:
        ax.text(0.5, 0.5,
                "Sin lb_history/ub_history\n(solo disponible para la lagrangiana)",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        return ax

    iteraciones = range(len(lb))
    ax.plot(iteraciones, lb, linewidth=1.5, color="#2980b9", label="LB (cota inferior)")
    ax.plot(iteraciones, ub, linewidth=1.5,
            color=estilo.PALETA_METODOS["lagrangiana"], label="UB (cota superior)")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Coste (€)")
    ax.set_title("Convergencia de la relajación lagrangiana")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
