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

# Muestra de tamaños para VISUALES POR INSTANCIA (mapas de solución, convergencia
# individual): única fuente de verdad de la política de muestra del plan (D9).
# Los cuadernos por método pintan un mapa por tamaño solo para RADIOS_MUESTRA ∩
# {tamaños con artefacto en disco}; las TABLAS y los GRÁFICOS AGREGADOS
# (escalabilidad) siguen usando TODOS los tamaños, no esta muestra.
# Criterio: 500 (mínimo legible) · 800/1000 (interior) · 1500 (máximo de memoria).
RADIOS_MUESTRA = [500, 800, 1000, 1500]

# Estado de OPTIMALIDAD de Gurobi por código de ``status`` (plan D10). Solo se mapean
# los estados que el análisis necesita distinguir. La ETIQUETA la decide el ``status``
# CRUDO (entero de ``Model.Status``), NUNCA el gap redondeado (un gap ≈ 0 puede venir de
# una solución NO demostrada óptima). estilo.py no importa gurobipy: solo mapea enteros.
#   2  → óptimo        (OPTIMAL, óptimo demostrado)
#   9  → time-limit    (TIME_LIMIT, censura por límite de tiempo: la meseta de 4 h)
#   11 → interrumpido  (INTERRUPTED, interrupción externa ANTES del límite)
# Este mapeo es la ÚNICA fuente; el módulo agnóstico ``comparativas`` NO lo conoce.
ESTADO_GUROBI = {2: "óptimo", 9: "time-limit", 11: "interrumpido"}

# Aspecto del marcador por estado en los gráficos (única fuente del estilo visual): el
# cuaderno pasa estos marcadores a ``comparativas.grafico_escalabilidad(resaltar=...)``.
MARCADOR_ESTADO = {"time-limit": "s", "interrumpido": "^"}


def nombre_tipo(k: int) -> str:
    """Etiqueta del tipo de residuo ``k`` (con respaldo genérico si no existe)."""
    return TIPOS_RESIDUO.get(k, f"tipo {k}")


def estado_gurobi(status: int) -> str:
    """Etiqueta de optimalidad del código de ``status`` de Gurobi (plan D10).

    Devuelve la etiqueta mapeada (``"óptimo"`` / ``"time-limit"`` / ``"interrumpido"``)
    o, si el código no está en ``ESTADO_GUROBI``, el código crudo (``"status <n>"``).
    NUNCA asume óptimo para un código desconocido (p. ej. 13 = SUBOPTIMAL)."""
    return ESTADO_GUROBI.get(status, f"status {status}")
