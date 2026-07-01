"""
(De)serialización sin pérdida de soluciones a JSON.

Sustituye a `pickle` como formato de persistencia de las soluciones del problema
(exacto, lagrangiana y —en el futuro— greedy/tabú). El objetivo es doble:

  1. Eliminar la fragilidad del pickle (acoplamiento a versiones de Python/numpy).
  2. Producir un fichero legible, anidado y autodescriptivo, alineado con la
     convención del módulo C++ de la metaheurística
     (`src/metaheuristica/src/io.cpp`), donde las estructuras indexadas por dos
     índices se escriben anidadas `{a: {b: v}}` con claves string.

Convención de serialización (espejo de io.cpp):
  - Estructuras con CLAVE-TUPLA del pickle (x[(j,k)], w[(j,k)], y_assign[(i,k)])
    se serializan ANIDADAS: {"j": {"k": v}}.
  - La estructura z[j] (un solo índice) se serializa PLANA: {"j": v}.
  - El resto de campos (escalares, históricos) se guardan tal cual.

La reconstrucción (módulo `carga`) devuelve estructuras IDÉNTICAS a las que daba
el pickle, de modo que el código consumidor no nota el cambio.

Este módulo NO importa gurobipy ni osmnx: persistir/cargar soluciones es
independiente del solver y de la cartografía.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import numpy as _np
    _ESCALARES_NUMPY: tuple = (_np.generic,)
except ImportError:                      # numpy es opcional para la carga
    _ESCALARES_NUMPY = ()


# Claves estructurales de una solución y cómo se serializan.
_CLAVES_ANIDADAS = ("x", "w", "y_assign")   # clave-tupla (a, b) → {"a": {"b": v}}
_CLAVE_PLANA = "z"                          # clave simple    a   → {"a": v}


def _nativo(v: Any) -> Any:
    """Convierte escalares de numpy a tipos nativos de Python, recursivamente
    sobre listas y diccionarios. Deja intacto lo que ya es nativo.

    La conversión numpy.float64 → float es por VALOR (misma representación
    IEEE-754 de doble precisión): no hay pérdida numérica."""
    if _ESCALARES_NUMPY and isinstance(v, _ESCALARES_NUMPY):
        return v.item()
    if isinstance(v, list):
        return [_nativo(x) for x in v]
    if isinstance(v, dict):
        return {k: _nativo(x) for k, x in v.items()}
    return v


def anidar(d: dict) -> dict:
    """{(a, b): v} → {"a": {"b": v}} (claves string, valores nativos).

    Preserva exactamente qué pares (a, b) están presentes: admite diccionario
    disperso y diccionario vacío (→ {})."""
    salida: dict = {}
    for (a, b), v in d.items():
        salida.setdefault(str(a), {})[str(b)] = _nativo(v)
    return salida


def aplanar_str(d: dict) -> dict:
    """{a: v} → {"a": v} (claves string, valores nativos)."""
    return {str(a): _nativo(v) for a, v in d.items()}


def desanidar(o: dict, tipo_val=int) -> dict:
    """Inverso de `anidar`: {"a": {"b": v}} → {(int(a), int(b)): tipo_val(v)}."""
    salida: dict = {}
    for a, sub in o.items():
        for b, v in sub.items():
            salida[(int(a), int(b))] = tipo_val(v)
    return salida


def desaplanar(o: dict, tipo_clave=int, tipo_val=int) -> dict:
    """Inverso de `aplanar_str`: {"a": v} → {tipo_clave(a): tipo_val(v)}."""
    return {tipo_clave(a): tipo_val(v) for a, v in o.items()}


def documento_solucion(sol: dict, metodo: str) -> dict:
    """Construye el documento JSON (anidado, sin pérdida) de una solución.

    Recorre TODAS las claves de `sol` sin omitir ninguna:
      - x, w, y_assign  → anidadas {"j": {"k": v}}
      - z               → plana {"j": v}
      - resto (cost, gap, históricos, ...) → tal cual (coercidos a nativo)

    Añade una clave `metodo` para trazabilidad y autodetección en la carga. Esa
    clave es el ÚNICO campo extra respecto del pickle; la carga la descarta para
    devolver un diccionario idéntico al original.
    """
    doc: dict = {"metodo": metodo}
    for clave, valor in sol.items():
        if clave in _CLAVES_ANIDADAS:
            doc[clave] = anidar(valor)
        elif clave == _CLAVE_PLANA:
            doc[clave] = aplanar_str(valor)
        else:
            doc[clave] = _nativo(valor)
    return doc


def _default_json(o: Any) -> Any:
    """Red de seguridad para json.dump: coacciona escalares numpy que se hayan
    podido colar; cualquier otra cosa no serializable lanza TypeError (en vez de
    persistir algo con pérdida silenciosa)."""
    if _ESCALARES_NUMPY and isinstance(o, _ESCALARES_NUMPY):
        return o.item()
    raise TypeError(f"No serializable a JSON: {type(o)}")


def guardar_documento(doc: dict, path: str) -> None:
    """Escribe un documento de solución a `path` en JSON indentado (UTF-8),
    creando el directorio padre si hace falta."""
    carpeta = os.path.dirname(path)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False, default=_default_json)
        f.write("\n")
