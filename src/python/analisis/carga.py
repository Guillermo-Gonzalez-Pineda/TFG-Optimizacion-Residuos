"""
Carga de soluciones desde JSON, reconstruyendo en memoria las MISMAS estructuras
que entregaba el pickle (diccionarios con clave-tupla incluidos).

Independiente del solver: NO importa gurobipy ni modelo_exacto. Sirve a cualquier
método (exacto, lagrangiana y futuros greedy/tabú con el mismo esquema de campos).
"""

from __future__ import annotations

import json

from .serializacion import (
    desanidar,
    desaplanar,
    _CLAVES_ANIDADAS,
    _CLAVE_PLANA,
)


def cargar_solucion(path: str) -> dict:
    """Lee un documento de solución JSON y reconstruye el diccionario en el
    formato histórico del pickle:

      - x, w, y_assign  → {(j, k): int}
      - z               → {j: int}
      - resto de campos → tal cual (escalares, históricos)

    La clave auxiliar `metodo` (metadato de escritura) se descarta para que el
    diccionario devuelto sea idéntico al que producía el pickle.
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    sol: dict = {}
    for clave, valor in doc.items():
        if clave == "metodo":
            continue
        if clave in _CLAVES_ANIDADAS:
            sol[clave] = desanidar(valor)
        elif clave == _CLAVE_PLANA:
            sol[clave] = desaplanar(valor)
        else:
            sol[clave] = valor
    return sol


def cargar_solucion_exacta(path: str) -> dict:
    """Carga una solución exacta. Devuelve el mismo diccionario que el pickle:
    z, x, w, y_assign (claves-tupla) + cost, gap_gurobi, runtime, status."""
    return cargar_solucion(path)


def cargar_solucion_lagrangiana(path: str) -> dict:
    """Carga una solución lagrangiana. Devuelve el mismo diccionario que el
    pickle: z, x, w, y_assign (claves-tupla) + cost, best_lb, best_ub, gap,
    lb_history, ub_history y demás metadatos/históricos."""
    return cargar_solucion(path)
