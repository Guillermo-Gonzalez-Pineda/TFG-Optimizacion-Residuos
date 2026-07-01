"""
Registro de métodos y esquema de rutas de salida — ÚNICA fuente de verdad.

Centraliza dónde vive cada artefacto (soluciones, instancias, grafos, figuras),
de modo que ni los cuadernos ni los módulos de análisis tengan rutas relativas
embebidas (riesgo de dependencia del directorio de trabajo señalado en el plan).

ESQUEMA NUEVO (sección 3 del plan):
    output/<metodo>/solucion_<adjetivo>_<tam>m.json

donde <metodo> es el directorio (p. ej. "exacto") y <adjetivo> es el del nombre
de fichero (p. ej. "exacta"). El registro resuelve los desajustes que el plan
marca como riesgo: exacto→exacta y metaheuristica→tabu.

⚠️  IMPORTANTE: este módulo SOLO define el esquema NUEVO. Los ficheros físicos
todavía están en el esquema VIEJO (output/<metodo>_<tam>m/solucion_<adjetivo>.json,
p. ej. output/exacto_1500m/solucion_exacta.json). El movimiento físico de los
ficheros al esquema nuevo es la FASE E del plan, posterior a esta. Hasta entonces,
`tamaños_disponibles()` (que globa el esquema nuevo) devolverá listas vacías para
los métodos cuyos ficheros aún no se hayan migrado.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Metodo:
    """Un método de resolución y sus dos nombres:

    - ``clave``: nombre del DIRECTORIO de salida (p. ej. ``"exacto"``).
    - ``adjetivo``: adjetivo usado en el NOMBRE DE FICHERO (p. ej. ``"exacta"``),
      que no siempre coincide con la clave (exacto→exacta, metaheuristica→tabu).
    """

    clave: str
    adjetivo: str


# Registro de métodos: clave de directorio → Metodo.
REGISTRO: dict[str, Metodo] = {
    "exacto":         Metodo("exacto",         "exacta"),
    "lagrangiana":    Metodo("lagrangiana",    "lagrangiana"),
    "greedy":         Metodo("greedy",         "greedy"),
    "metaheuristica": Metodo("metaheuristica", "tabu"),
}

# Subdirectorios estándar (relativos a la raíz del repo).
_DIR_SALIDA = "output"
_DIR_DATOS = "data/processed"
_DIR_FIGURAS = "figuras"


def raiz_repo() -> Path:
    """Devuelve la raíz del repositorio como ruta ABSOLUTA, subiendo desde este
    fichero hasta encontrar el directorio ``.git``.

    Ancla todas las rutas a una base fija e independiente del directorio de
    trabajo (cwd), eliminando el riesgo de árboles de salida fantasma que el plan
    documenta (p. ej. el ``src/python/output/`` creado al ejecutar desde el
    directorio equivocado)."""
    for carpeta in Path(__file__).resolve().parents:
        if (carpeta / ".git").exists():
            return carpeta
    raise RuntimeError(
        "No se encontró la raíz del repositorio (.git) por encima de "
        f"{Path(__file__).resolve()}"
    )


def _metodo(clave: str) -> Metodo:
    """Resuelve la clave de método contra el registro, con error claro."""
    try:
        return REGISTRO[clave]
    except KeyError:
        disponibles = ", ".join(sorted(REGISTRO))
        raise KeyError(
            f"Método desconocido: {clave!r}. Métodos válidos: {disponibles}."
        ) from None


def dir_metodo(metodo: str) -> Path:
    """Directorio de salida de un método: ``output/<metodo>/``."""
    return raiz_repo() / _DIR_SALIDA / _metodo(metodo).clave


def ruta_solucion_json(metodo: str, tam: int) -> Path:
    """Ruta de la solución completa (esquema NUEVO):
    ``output/<metodo>/solucion_<adjetivo>_<tam>m.json``."""
    m = _metodo(metodo)
    return dir_metodo(metodo) / f"solucion_{m.adjetivo}_{tam}m.json"


def ruta_instancia(tam: int) -> Path:
    """Ruta de la instancia: ``data/processed/instancia_laguna_<tam>m.json``."""
    return raiz_repo() / _DIR_DATOS / f"instancia_laguna_{tam}m.json"


def ruta_grafo(tam: int) -> Path:
    """Ruta del grafo de calles: ``data/processed/graph_<tam>m.graphml``."""
    return raiz_repo() / _DIR_DATOS / f"graph_{tam}m.graphml"


def ruta_buildings(tam: int) -> Path:
    """Ruta de los edificios: ``data/processed/buildings_<tam>m.geojson``."""
    return raiz_repo() / _DIR_DATOS / f"buildings_{tam}m.geojson"


def ruta_figura(nombre: str) -> Path:
    """Ruta de una figura de salida: ``output/figuras/<nombre>.png``."""
    return raiz_repo() / _DIR_SALIDA / _DIR_FIGURAS / f"{nombre}.png"


def tamaños_disponibles(metodo: str) -> list[int]:
    """Tamaños (en metros) con solución disponible para ``metodo``, en orden
    ascendente. Globa el esquema NUEVO ``output/<metodo>/solucion_<adjetivo>_*m.json``.

    Nota: mientras los ficheros sigan en el esquema viejo (antes de la Fase E),
    esta función devuelve ``[]`` para los métodos aún no migrados."""
    m = _metodo(metodo)
    patron = re.compile(rf"^solucion_{re.escape(m.adjetivo)}_(\d+)m$")
    tams: list[int] = []
    for ruta in dir_metodo(metodo).glob(f"solucion_{m.adjetivo}_*m.json"):
        casa = patron.match(ruta.stem)
        if casa:
            tams.append(int(casa.group(1)))
    return sorted(tams)
