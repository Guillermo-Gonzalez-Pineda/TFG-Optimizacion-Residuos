"""
Constantes de estilo compartidas por las visualizaciones del análisis.

Módulo de CONSTANTES PURAS: cero dependencias de proyecto y, deliberadamente,
NINGÚN import de matplotlib. Los nombres de mapa de color se guardan como
*strings* (p. ej. "YlGnBu"): así este módulo no arrastra matplotlib, y quien
dibuje hará `plt.get_cmap(estilo.CMAP_DEMANDA)` o pasará el string a `cmap=...`.
"""

from __future__ import annotations


# Nombres de los tipos de residuo (índice k → etiqueta).
TIPOS_RESIDUO = {0: "Orgánica", 1: "Resto", 2: "Reciclable", 3: "Peligrosos"}

# Estilo de tablas (unifica el de los cuadernos 02c3 y 03c5).
COLOR_CABECERA = "#2c3e50"
COLOR_TEXTO_CABECERA = "white"
COLOR_FILA_PAR = "#ecf0f1"
COLOR_FILA_IMPAR = "white"

# Mapas de color (como string, ver nota de cabecera: estilo.py no importa mpl).
CMAP_DEMANDA = "YlGnBu"   # gradiente de demanda (habitantes por edificio/punto)
CMAP_CARGA = "YlOrRd"     # gradiente de carga (residuo asignado a un punto)

# Color fijo por método para gráficos comparativos.
# Las claves DEBEN coincidir con las del REGISTRO de rutas.py (exacto,
# lagrangiana, greedy, metaheuristica). Se verifica en tests/verificar_comparativas.py
# (estilo.py no importa rutas para mantenerse sin dependencias de proyecto).
PALETA_METODOS = {
    "exacto":         "#2c3e50",
    "lagrangiana":    "#e67e22",
    "greedy":         "#27ae60",
    "metaheuristica": "#8e44ad",
}


def nombre_tipo(k: int) -> str:
    """Etiqueta del tipo de residuo ``k`` (con respaldo genérico si no existe)."""
    return TIPOS_RESIDUO.get(k, f"tipo {k}")
