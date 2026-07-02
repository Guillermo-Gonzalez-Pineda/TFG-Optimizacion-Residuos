"""
Paquete de análisis: carga, métricas, visualización y comparativas de las
soluciones del problema de localización de contenedores de residuos.

IMPORTANTE: este `__init__` solo importa módulos LIGEROS (serialización y carga),
sin dependencias de gurobipy ni osmnx, para que persistir/cargar soluciones no
arrastre el solver ni la cartografía. Los módulos pesados (p. ej. `mapas`, con
osmnx/geopandas) deben importarse explícitamente: `from analisis.mapas import ...`.
"""

from __future__ import annotations

from .serializacion import (
    documento_solucion,
    guardar_documento,
    anidar,
    aplanar_str,
    desanidar,
    desaplanar,
)
from .carga import (
    cargar_solucion,
    cargar_solucion_exacta,
    cargar_solucion_lagrangiana,
)
from .rutas import (
    Metodo,
    REGISTRO,
    raiz_repo,
    ruta_solucion_json,
    ruta_instancia,
    ruta_grafo,
    ruta_buildings,
    ruta_figura,
    tamaños_disponibles,
)
from .metricas import (
    puntos_abiertos,
    n_puntos_abiertos,
    total_bins,
    bins_por_tipo,
    coste,
    gap,
    desglose_coste,
    violaciones_capacidad,
    demanda_por_punto,
    resumen,
    outliers_demanda_iqr,
    cobertura_por_tipo,
    candidatos_reales_vs_artificiales,
    resumen_distancias,
)
# `factibilidad` es solver-free y geo-free (compute_demand se importa perezosamente):
# seguro re-exportarlo.
from . import factibilidad
from .factibilidad import candidatos_validos, verificar_hdm

# `estilo` son constantes puras (sin matplotlib): es seguro re-exportarlo.
from . import estilo
from .estilo import (
    TIPOS_RESIDUO,
    PALETA_METODOS,
    RADIOS_MUESTRA,
    ESTADO_GUROBI,
    MARCADOR_ESTADO,
    nombre_tipo,
    estado_gurobi,
)

# NOTA: `comparativas` NO se importa aquí a propósito — arrastraría matplotlib/pandas
# a cualquier `import analisis`. Úsese bajo demanda: `from analisis.comparativas import ...`.
# `mapas` (osmnx/geopandas) tampoco debe importarse nunca a nivel de paquete.

__all__ = [
    # serialización
    "documento_solucion",
    "guardar_documento",
    "anidar",
    "aplanar_str",
    "desanidar",
    "desaplanar",
    # carga
    "cargar_solucion",
    "cargar_solucion_exacta",
    "cargar_solucion_lagrangiana",
    # rutas
    "Metodo",
    "REGISTRO",
    "raiz_repo",
    "ruta_solucion_json",
    "ruta_instancia",
    "ruta_grafo",
    "ruta_buildings",
    "ruta_figura",
    "tamaños_disponibles",
    # métricas
    "puntos_abiertos",
    "n_puntos_abiertos",
    "total_bins",
    "bins_por_tipo",
    "coste",
    "gap",
    "desglose_coste",
    "violaciones_capacidad",
    "demanda_por_punto",
    "resumen",
    "outliers_demanda_iqr",
    "cobertura_por_tipo",
    "candidatos_reales_vs_artificiales",
    "resumen_distancias",
    # factibilidad (verificación de restricciones del HDM, solver-free)
    "factibilidad",
    "candidatos_validos",
    "verificar_hdm",
    # estilo (constantes puras)
    "estilo",
    "TIPOS_RESIDUO",
    "PALETA_METODOS",
    "RADIOS_MUESTRA",
    "ESTADO_GUROBI",
    "MARCADOR_ESTADO",
    "nombre_tipo",
    "estado_gurobi",
]
