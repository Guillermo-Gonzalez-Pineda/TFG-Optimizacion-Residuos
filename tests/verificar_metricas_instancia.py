"""
Verificación ejecutable de las MÉTRICAS DE INSTANCIA geo-free (Fase D):
cobertura_por_tipo, outliers_demanda_iqr, candidatos_reales_vs_artificiales,
resumen_distancias. Patrón del repo (sin pytest).

Uso (desde la raíz del repo):
    venv/bin/python tests/verificar_metricas_instancia.py
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from instancia import load_instance

from analisis import metricas, rutas

TAM = 1500
fallos: list[str] = []


def marca(ok: bool) -> str:
    return "✅" if ok else "❌"


def main() -> None:
    inst = load_instance(str(rutas.ruta_instancia(TAM)))

    print("=" * 74)
    print(f"  MÉTRICAS DE INSTANCIA — verificación  (tam = {TAM} m)")
    print("=" * 74)

    # ── A. candidatos reales vs artificiales ──
    cra = metricas.candidatos_reales_vs_artificiales(inst)
    okA = cra["reales"] + cra["artificiales"] == cra["total"] == inst.n_candidates
    print(f"\n  A. candidatos: total={cra['total']} reales={cra['reales']} "
          f"artificiales={cra['artificiales']}  (n_candidates={inst.n_candidates})  {marca(okA)}")
    if not okA:
        fallos.append("A candidatos")

    # ── B. cobertura_por_tipo: monotonía en el radio + bounds ──
    cob = metricas.cobertura_por_tipo(inst)
    cov = inst.params.coverage_radius
    orden = sorted(inst.K, key=lambda k: cov[k])
    print("\n  B. cobertura (tipos ordenados por radio ascendente):")
    print(f"       {'k':>2} {'radio':>7} {'sin_cob':>8} {'acc_min':>8} {'acc_max':>8}  ok")
    prev = None
    okB = True
    for k in orden:
        pt = cob["por_tipo"][k]
        s = pt["edificios_sin_cobertura"]
        bounds = pt["acc_min"] >= 0 and pt["acc_max"] <= inst.n_candidates
        mono = prev is None or s <= prev
        ok_k = bounds and mono
        okB = okB and ok_k
        print(f"       {k:>2} {cov[k]:>6.0f}m {s:>8} {pt['acc_min']:>8} {pt['acc_max']:>8}  {marca(ok_k)}")
        prev = s
    print(f"       → monotonía no creciente + acc_min≥0 + acc_max≤n_candidates: {marca(okB)}")
    if not okB:
        fallos.append("B monotonía/bounds cobertura")

    # ── C. outliers_demanda_iqr: partición + conservación + pct∈[0,1] ──
    out = metricas.outliers_demanda_iqr(inst)
    idx_set = set(out["indices"])
    part_ok = (all(inst.I[i].h_i > out["umbral"] for i in out["indices"])
               and all(inst.I[i].h_i <= out["umbral"] for i in inst.I if i not in idx_set))
    cons_ok = abs(out["demanda_total"] - inst.total_population) < 1e-6
    pct_ok = 0.0 <= out["pct_demanda_outliers"] <= 1.0
    okC = part_ok and cons_ok and pct_ok
    print(f"\n  C. outliers: umbral={out['umbral']:.2f}  n_outliers={out['n_outliers']}  "
          f"pct={out['pct_demanda_outliers']:.4f}")
    print(f"       partición h_i≷umbral {marca(part_ok)} | "
          f"demanda_total={out['demanda_total']:.6f} vs total_population={inst.total_population:.6f} "
          f"{marca(cons_ok)} | 0≤pct≤1 {marca(pct_ok)}")
    if not okC:
        fallos.append("C outliers")

    # ── D. resumen_distancias: orden de stats + n + conteos ──
    rd = metricas.resumen_distancias(inst)
    okD = (rd["min"] <= rd["media"] <= rd["max"]
           and rd["min"] <= rd["mediana"] <= rd["max"]
           and rd["n"] == inst.n_dijkstra_connections
           and all(rd[x] <= rd["n"] for x in ("n_cero", "n_sub5", "n_sobre_cutoff")))
    print(f"\n  D. distancias (cutoff Dijkstra = {inst.dijkstra_radius_m} m):")
    print(f"       n={rd['n']} (meta {inst.n_dijkstra_connections})  "
          f"min={rd['min']:.1f} media={rd['media']:.1f} mediana={rd['mediana']:.1f} max={rd['max']:.1f}")
    print(f"       n_cero={rd['n_cero']}  n_sub5={rd['n_sub5']}  n_sobre_cutoff={rd['n_sobre_cutoff']}  {marca(okD)}")
    if not okD:
        fallos.append("D distancias")

    # ── E. REGRESIÓN: verificar_metricas.py sigue 15/15 ──
    print("\n  E. REGRESIÓN verificar_metricas.py:")
    ruta_reg = os.path.join(os.path.dirname(__file__), "verificar_metricas.py")
    r = subprocess.run([sys.executable, ruta_reg], capture_output=True, text=True)
    for linea in r.stdout.strip().splitlines()[-2:]:
        print("       " + linea)
    okE = r.returncode == 0
    print(f"       returncode={r.returncode}  {marca(okE)}")
    if not okE:
        fallos.append("E regresión verificar_metricas")

    print("\n" + "=" * 74)
    if fallos:
        print(f"  ❌ FALLOS ({len(fallos)}):")
        for f in fallos:
            print(f"      - {f}")
        sys.exit(1)
    print("  ✅ TODO OK — métricas de instancia verificadas (A–E).")
    print("=" * 74)


if __name__ == "__main__":
    main()
