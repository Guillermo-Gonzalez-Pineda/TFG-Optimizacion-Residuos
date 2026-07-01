"""
Visualización cartográfica de instancias y soluciones (Fase C.2).

ÚNICO módulo del paquete ``analisis`` autorizado a importar la pila geoespacial
pesada (osmnx, geopandas, shapely). Por eso ``__init__`` NO lo re-exporta: quien
lo necesite hace ``from analisis.mapas import ...`` y solo entonces se paga el
coste de importar osmnx/geopandas. PROHIBIDO importar gurobipy (ni transitivamente):
aquí se dibujan soluciones YA resueltas, nunca se resuelve nada.

Construye sobre la API real de los módulos ligeros:
  - ``estilo``   → colores/cmaps y ``TIPOS_RESIDUO`` (cero literales de color aquí).
  - ``metricas`` → ``puntos_abiertos`` y ``demanda_por_punto`` (no se duplica lógica).
  - ``rutas``    → rutas de grafo/edificios/figuras ancladas a la raíz del repo.
  - ``instancia``→ solo para type-hints (las coordenadas de I/J llegan en ``inst``).

Tres decisiones de diseño, comentadas donde ocurren:
  1. El GeoJSON de edificios se carga A MANO (``json.load`` + ``shapely.shape``),
     sin fiona: fiona no está instalado en este entorno (choca de versión con
     Python 3.14) y geopandas 1.x delega en pyogrio; cargándolo a mano no
     dependemos de NINGÚN driver OGR. Ver ``_cargar_edificios_geojson``.
  2. ``emparejar_demanda`` calcula centroides sobre EPSG:4326, lo que dispara el
     aviso "geometry is in a geographic CRS" de geopandas. Aquí es un FALSO
     POSITIVO (solo medimos cercanía, no áreas; edificios e instancia comparten
     CRS) y se suprime SOLO ese aviso, sin reproyectar (reproyectar cambiaría el
     umbral en grados). Ver ``emparejar_demanda``.
  3. ``mapa_solucion`` es UNA sola función (sustituye a los antiguos
     ``plot_exact_solution`` + ``plot_rich_solution`` + ``plot_rich_solution_lagrangian``):
     recibe una ``Solucion`` YA NORMALIZADA (mismo dict para exacto/lagrangiana/…),
     así que no hay nada específico de método que ramificar. Ver ``mapa_solucion``.
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from shapely.geometry import shape

from . import estilo, metricas, rutas

if TYPE_CHECKING:                       # coordenadas de I/J llegan por parámetro:
    from instancia import Instance      # no acoplamos el import en tiempo de ejecución


# Aviso de geopandas al pedir .centroid sobre un CRS geográfico (ver decisión 2).
_AVISO_CENTROID = "Geometry is in a geographic CRS"

# Umbral de emparejamiento edificio↔polígono, en GRADOS (mismo CRS, EPSG:4326).
_UMBRAL_GRADOS = 0.0005


# ─────────────────────────── carga de fondo ────────────────────────────────

def _cargar_edificios_geojson(path) -> gpd.GeoDataFrame:
    """Carga un GeoJSON de edificios SIN fiona (decisión 1).

    ``json.load`` + ``shapely.shape`` reconstruyen las geometrías sin tocar
    ningún driver OGR (fiona no está instalado; geopandas 1.x usaría pyogrio).
    Conserva las propiedades como columnas del ``GeoDataFrame``."""
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    geoms = [shape(ft["geometry"]) for ft in feats]
    props = [ft.get("properties", {}) or {} for ft in feats]
    return gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")


def cargar_fondo(tam: int):
    """Devuelve ``(grafo_calles, gdf_edificios)`` para el tamaño ``tam``.

    - ``grafo``: red de calles vía ``ox.load_graphml`` (GraphML, sin OGR).
    - ``gdf_edificios``: polígonos de edificios (GeoJSON a mano, ver decisión 1)."""
    grafo = ox.load_graphml(rutas.ruta_grafo(tam))
    gdf_edificios = _cargar_edificios_geojson(rutas.ruta_buildings(tam))
    return grafo, gdf_edificios


# ─────────────────────────── primitivas de dibujo ──────────────────────────

def _plot_calles(ax, grafo, *, linewidth: float = 0.5, alpha: float = 0.35):
    """Pinta las aristas del grafo en ``ax`` (helper interno para no duplicar la
    conversión grafo→gdf entre ``dibujar_calles`` y los mapas compuestos)."""
    aristas = ox.graph_to_gdfs(grafo, nodes=False)
    aristas.plot(ax=ax, color=estilo.COLOR_CABECERA,
                 linewidth=linewidth, alpha=alpha, zorder=1)
    return ax


def dibujar_calles(ax, tam: int):
    """Dibuja la red de calles del tamaño ``tam`` sobre ``ax``."""
    grafo = ox.load_graphml(rutas.ruta_grafo(tam))
    return _plot_calles(ax, grafo)


def dibujar_edificios(ax, gdf: gpd.GeoDataFrame, inst: "Instance" | None = None):
    """Dibuja las huellas de los edificios.

    - ``inst is None``: relleno neutro (solo contexto).
    - ``inst`` dado: colorea por población ``h_i`` (vía ``emparejar_demanda``),
      con ``estilo.CMAP_DEMANDA`` y barra de color; los polígonos sin edificio
      emparejado quedan en el gris neutro de ``estilo``."""
    if inst is None:
        gdf.plot(ax=ax, facecolor=estilo.COLOR_FILA_PAR,
                 edgecolor=estilo.COLOR_CABECERA, linewidth=0.2, alpha=0.6, zorder=2)
        return ax

    gdf_h = emparejar_demanda(gdf, inst)
    gdf_h.plot(ax=ax, column="h_i", cmap=estilo.CMAP_DEMANDA,
               edgecolor=estilo.COLOR_CABECERA, linewidth=0.2, zorder=2,
               legend=True, legend_kwds={"label": "Población h_i (hab)", "shrink": 0.6},
               missing_kwds={"color": estilo.COLOR_FILA_PAR})
    return ax


def emparejar_demanda(gdf_edificios: gpd.GeoDataFrame, inst: "Instance",
                      umbral: float = _UMBRAL_GRADOS) -> gpd.GeoDataFrame:
    """Empareja cada edificio-demanda de ``inst.I`` con su polígono más cercano.

    Para cada ``inst.I[i]`` busca el centroide de polígono más próximo (distancia
    euclídea en grados) y, si cae por debajo de ``umbral`` (0.0005° ≈ pocos metros
    a esta latitud), le adjudica su población ``h_i``. Devuelve una COPIA del gdf
    con una columna ``h_i`` (NaN en los polígonos sin edificio emparejado).

    El ``.centroid`` sobre EPSG:4326 dispara el aviso de CRS geográfico de
    geopandas; aquí es un falso positivo (medimos cercanía, no áreas; mismo CRS)
    y lo suprimimos SOLO a él, sin reproyectar (decisión 2 de la cabecera)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_AVISO_CENTROID, category=UserWarning)
        centroides = gdf_edificios.geometry.centroid

    cx = centroides.x.to_numpy()
    cy = centroides.y.to_numpy()
    umbral2 = umbral * umbral

    h = np.full(len(gdf_edificios), np.nan)
    for edif in inst.I.values():
        d2 = (cx - edif.longitude) ** 2 + (cy - edif.latitude) ** 2
        idx = int(np.argmin(d2))
        if d2[idx] < umbral2:
            h[idx] = edif.h_i

    gdf = gdf_edificios.copy()
    gdf["h_i"] = h
    return gdf


# ─────────────────────────── mapas compuestos ──────────────────────────────

def _nuevo_ax(ax, figsize):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _tam_de(inst: "Instance", tam: int | None) -> int:
    """Tamaño efectivo: el explícito o, si no, el radio OSM de la instancia."""
    return inst.osm_radius_m if tam is None else tam


def mapa_instancia(inst: "Instance", tam: int | None = None, ax=None):
    """Edificios + candidatos sobre la red de calles (planteamiento del problema)."""
    tam = _tam_de(inst, tam)
    ax = _nuevo_ax(ax, (10, 10))
    grafo, gdf = cargar_fondo(tam)
    _plot_calles(ax, grafo)
    dibujar_edificios(ax, gdf)                       # neutro (sin color de demanda)
    ax.scatter([c.longitude for c in inst.J.values()],
               [c.latitude for c in inst.J.values()],
               marker="^", s=22, color=estilo.COLOR_CABECERA, zorder=5)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=estilo.COLOR_FILA_PAR,
               markeredgecolor=estilo.COLOR_CABECERA, label=f"Edificios ({len(inst.I)})"),
        Line2D([0], [0], marker="^", linestyle="", color=estilo.COLOR_CABECERA,
               label=f"Candidatos ({len(inst.J)})"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(f"Instancia · {inst.study_case} · {tam} m", fontsize=12, fontweight="bold")
    ax.set_axis_off()
    return ax


def mapa_demanda(inst: "Instance", tam: int | None = None, ax=None):
    """Edificios coloreados por población ``h_i`` sobre la red de calles."""
    tam = _tam_de(inst, tam)
    ax = _nuevo_ax(ax, (10, 10))
    grafo, gdf = cargar_fondo(tam)
    _plot_calles(ax, grafo)
    dibujar_edificios(ax, gdf, inst)                 # color ∝ h_i
    ax.set_title(f"Demanda por edificio (h_i) · {inst.study_case} · {tam} m",
                 fontsize=12, fontweight="bold")
    ax.set_axis_off()
    return ax


def mapa_solucion(sol: dict, inst: "Instance" | None = None, rich: bool = True, ax=None):
    """Mapa de una solución. FUNCIÓN ÚNICA para todo método (ver decisión 3).

    Requiere ``inst``: las coordenadas de los puntos ``j`` viven en ``inst.J`` (la
    solución solo trae índices), y la demanda por punto necesita ``inst.I``.

    - ``rich=False``: puntos abiertos sobre las calles.
    - ``rich=True`` : calles de fondo; líneas de asignación (``y_assign``, k=0);
      candidatos cerrados; y puntos abiertos con TAMAÑO ∝ nº de contenedores y
      COLOR ∝ ``metricas.demanda_por_punto(sol, inst, 0)`` (cmap ``estilo.CMAP_DEMANDA``).
      La etiqueta del canal usa ``estilo.TIPOS_RESIDUO`` vía ``estilo.nombre_tipo``."""
    if inst is None:
        raise ValueError(
            "mapa_solucion necesita 'inst': las coordenadas de los puntos (inst.J) "
            "y la demanda (inst.I) no están en la solución normalizada."
        )
    tam = inst.osm_radius_m
    ax = _nuevo_ax(ax, (11, 11))
    dibujar_calles(ax, tam)

    abiertos = metricas.puntos_abiertos(sol)

    if not rich:
        ax.scatter([inst.J[j].longitude for j in abiertos],
                   [inst.J[j].latitude for j in abiertos],
                   s=45, color=estilo.COLOR_CABECERA, zorder=5)
        ax.set_title(f"Puntos abiertos ({len(abiertos)}) · {inst.study_case} · {tam} m",
                     fontsize=12, fontweight="bold")
        ax.set_axis_off()
        return ax

    # ── candidatos cerrados ──
    cerrados = [j for j, abierto in sol["z"].items() if abierto == 0]
    if cerrados:
        ax.scatter([inst.J[j].longitude for j in cerrados],
                   [inst.J[j].latitude for j in cerrados],
                   marker="x", s=12, color=estilo.COLOR_CABECERA,
                   alpha=0.35, linewidths=0.6, zorder=3)

    # ── líneas de asignación edificio→punto para el tipo k=0 ──
    segmentos = []
    for (i, k), j in sol["y_assign"].items():
        if k != 0 or j < 0:            # otro tipo, o asignación no factible (j<0)
            continue
        edif, punto = inst.I[i], inst.J[j]
        segmentos.append([(edif.longitude, edif.latitude),
                          (punto.longitude, punto.latitude)])
    if segmentos:
        ax.add_collection(LineCollection(
            segmentos, colors=estilo.COLOR_CABECERA, linewidths=0.3, alpha=0.12, zorder=2))

    # ── puntos abiertos: tamaño ∝ bins, color ∝ demanda(k=0) ──
    bins_por_punto: dict[int, int] = {}
    for (j, k), n in sol["x"].items():
        bins_por_punto[j] = bins_por_punto.get(j, 0) + n
    demanda = metricas.demanda_por_punto(sol, inst, 0)

    xs = [inst.J[j].longitude for j in abiertos]
    ys = [inst.J[j].latitude for j in abiertos]
    tam_marca = [30 + 12 * bins_por_punto.get(j, 0) for j in abiertos]
    color_dem = [demanda.get(j, 0.0) for j in abiertos]

    sc = ax.scatter(xs, ys, s=tam_marca, c=color_dem, cmap=estilo.CMAP_DEMANDA,
                    edgecolor=estilo.COLOR_CABECERA, linewidths=0.4, zorder=5)
    barra = ax.figure.colorbar(sc, ax=ax, shrink=0.6)
    barra.set_label(f"Demanda · {estilo.nombre_tipo(0)} (hab)")

    handles = [
        Line2D([0], [0], color=estilo.COLOR_CABECERA, alpha=0.5, linewidth=1.2,
               label=f"Asignación · {estilo.nombre_tipo(0)} (k=0)"),
        Line2D([0], [0], marker="x", linestyle="", color=estilo.COLOR_CABECERA,
               alpha=0.5, label=f"Candidatos cerrados ({len(cerrados)})"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=estilo.COLOR_FILA_IMPAR,
               markeredgecolor=estilo.COLOR_CABECERA,
               label=f"Puntos abiertos ({len(abiertos)}) · tamaño ∝ nº bins"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"Solución · {inst.study_case} · {tam} m  "
        f"({len(abiertos)} puntos, {sum(bins_por_punto.values())} contenedores)",
        fontsize=12, fontweight="bold")
    ax.set_axis_off()
    return ax
