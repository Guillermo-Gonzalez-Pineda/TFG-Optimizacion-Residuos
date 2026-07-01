"""
Migración PICKLE → JSON de las soluciones ya calculadas (sin pérdida, verificada).

Para cada pickle existente:
    output/exacto_*/solucion_exacta.pkl
    output/lagrangiana_*/solucion_lagrangiana.pkl

  1. carga el pickle,
  2. lo reescribe como JSON anidado (al lado, MISMO nombre con extensión .json),
  3. recarga el JSON con `analisis.carga.cargar_solucion`,
  4. compara campo por campo contra el pickle original.

NO borra ningún pickle: solo escribe los .json al lado e informa. Si algún
artefacto no es idéntico tras el round-trip, lo marca como FALLO y detalla las
diferencias (y termina con código de salida 1).

Uso (desde la raíz del repo):
    python scripts/migrar_pickle_a_json.py
"""

from __future__ import annotations

import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from analisis.serializacion import documento_solucion, guardar_documento
from analisis.carga import cargar_solucion


# (método, patrón glob) de los artefactos a migrar.
PATRONES = [
    ("exacto",      "output/exacto_*/solucion_exacta.pkl"),
    ("lagrangiana", "output/lagrangiana_*/solucion_lagrangiana.pkl"),
]


def _comparar(orig: dict, recon: dict) -> list[str]:
    """Diferencias entre el pickle original y la reconstrucción desde JSON.
    Lista vacía ⇒ idénticos (migración sin pérdida).

    La igualdad es POR VALOR: un numpy.float64 del pickle y el float del JSON se
    consideran iguales (misma representación IEEE-754). Esa normalización de tipo
    es justamente el objetivo de eliminar el pickle/numpy."""
    problemas: list[str] = []

    claves_orig, claves_recon = set(orig), set(recon)
    if claves_orig != claves_recon:
        if claves_orig - claves_recon:
            problemas.append(f"  claves SOLO en pickle: {sorted(claves_orig - claves_recon)}")
        if claves_recon - claves_orig:
            problemas.append(f"  claves SOLO en JSON:   {sorted(claves_recon - claves_orig)}")
        return problemas

    for k in orig:
        a, b = orig[k], recon[k]
        if isinstance(a, dict):
            if set(a) != set(b):
                problemas.append(
                    f"  campo '{k}': claves de dict distintas "
                    f"(faltan {len(set(a) - set(b))}, sobran {len(set(b) - set(a))})"
                )
                continue
            difer = [(kk, a[kk], b[kk]) for kk in a if a[kk] != b[kk]]
            if difer:
                problemas.append(f"  campo '{k}': {len(difer)} valor(es) distinto(s); ej {difer[:3]}")
        elif isinstance(a, list):
            if len(a) != len(b):
                problemas.append(f"  campo '{k}': longitudes de lista {len(a)} vs {len(b)}")
                continue
            difer = [(i, a[i], b[i]) for i in range(len(a)) if a[i] != b[i]]
            if difer:
                problemas.append(f"  campo '{k}': {len(difer)} elemento(s) de lista distinto(s); ej {difer[:3]}")
        else:
            if a != b:
                problemas.append(f"  campo '{k}': escalar {a!r} != {b!r}")
    return problemas


def main() -> None:
    total = ok = fallos = 0
    artefactos_fallidos: list[str] = []

    print("=" * 70)
    print("  MIGRACIÓN PICKLE → JSON (sin pérdida, verificada por round-trip)")
    print("=" * 70)

    for metodo, patron in PATRONES:
        for pkl_path in sorted(glob.glob(patron)):
            total += 1
            json_path = pkl_path[:-len(".pkl")] + ".json"

            with open(pkl_path, "rb") as f:
                original = pickle.load(f)

            # Escribir el JSON al lado (el pickle no se toca)
            guardar_documento(documento_solucion(original, metodo), json_path)

            # Recargar el JSON y comparar contra el pickle original
            recon = cargar_solucion(json_path)
            problemas = _comparar(original, recon)

            if not problemas:
                ok += 1
                print(f"  ✅ {pkl_path}")
                print(f"       → {json_path}  ({len(original)} campos · round-trip idéntico)")
            else:
                fallos += 1
                artefactos_fallidos.append(pkl_path)
                print(f"  ❌ {pkl_path}  — {len(problemas)} diferencia(s):")
                for p in problemas:
                    print(p)

    print("\n" + "=" * 70)
    print(f"  RESULTADO: {total} artefactos | {ok} sin pérdida | {fallos} con diferencias")
    print("=" * 70)
    if fallos:
        print("  Artefactos con diferencias (ningún pickle se ha borrado):")
        for d in artefactos_fallidos:
            print(f"    - {d}")
        sys.exit(1)
    print("  Todos los pickle se han reescrito a JSON SIN PÉRDIDA.")
    print("  Los .pkl siguen intactos; su borrado es un paso manual posterior.")


if __name__ == "__main__":
    main()
