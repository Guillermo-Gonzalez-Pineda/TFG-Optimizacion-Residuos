"""
Verificación ejecutable de ``analisis.metricas.demanda_por_punto`` (completa la
Fase B). Patrón del repo (sin pytest, cf. verificar_metricas.py / verificar_refactor.py).

Sobre la solución EXACTA de 500m (totalmente asignada) comprueba:
  1. Conservación: para cada tipo k, Σ_j demanda == población total de la instancia
     (cada edificio asigna su tipo k a exactamente un punto en solución factible).
  2. Los puntos con demanda del tipo 0 son un subconjunto de los puntos abiertos.
  3. Top-3 (punto, demanda) como muestra legible.

Uso (desde la raíz del repo):
    venv/bin/python tests/verificar_demanda.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from instancia import load_instance

from analisis import metricas, rutas
from analisis.carga import cargar_solucion

TAM = 500
TOL = 1e-6


def main() -> None:
    raiz = rutas.raiz_repo()
    sol = cargar_solucion(raiz / "output" / f"exacto_{TAM}m" / "solucion_exacta.json")
    inst = load_instance(str(rutas.ruta_instancia(TAM)))

    fallos: list[str] = []

    print("=" * 70)
    print(f"  VERIFICACIÓN metricas.demanda_por_punto  (exacta {TAM}m)")
    print("=" * 70)

    # ── 1. Invariante de conservación por tipo k ──
    print(f"\n  1. Conservación (Σ_j demanda == total_population = "
          f"{inst.total_population:.6f}):")
    for k in range(4):
        total_k = sum(metricas.demanda_por_punto(sol, inst, k).values())
        ok = abs(total_k - inst.total_population) < TOL
        print(f"       k={k}: Σ demanda = {total_k:.6f}  vs  "
              f"{inst.total_population:.6f}  {'✅' if ok else '❌'}")
        if not ok:
            fallos.append(f"conservación k={k}: {total_k} != {inst.total_population}")

    # ── 2. keys(demanda k=0) ⊆ puntos_abiertos ──
    dpp0 = metricas.demanda_por_punto(sol, inst, 0)
    abiertos = set(metricas.puntos_abiertos(sol))
    claves = set(dpp0)
    subconj = claves <= abiertos
    print(f"\n  2. keys(demanda k=0) ⊆ puntos_abiertos: {'✅' if subconj else '❌'}")
    print(f"       |keys|={len(claves)}  |abiertos|={len(abiertos)}  "
          f"fuera={sorted(claves - abiertos)}")
    if not subconj:
        fallos.append(f"keys no ⊆ abiertos: {sorted(claves - abiertos)}")

    # ── 3. Top-3 (punto, demanda) ──
    top3 = sorted(dpp0.items(), key=lambda kv: kv[1], reverse=True)[:3]
    print("\n  3. Top-3 puntos por demanda (k=0):")
    for j, d in top3:
        print(f"       punto {j:>4} : {d:>10,.2f} hab")

    print("\n" + "=" * 70)
    if fallos:
        print(f"  ❌ FALLOS ({len(fallos)}):")
        for f in fallos:
            print(f"      - {f}")
        sys.exit(1)
    print("  ✅ TODO OK — demanda_por_punto verificada (conservación + subconjunto).")
    print("=" * 70)


if __name__ == "__main__":
    main()
