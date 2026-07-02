#include "tabu/oracle.hpp"

#ifdef ORACLE_CHECK
// ===========================================================================
//  ORÁCULO DE CONSISTENCIA (solo con -DORACLE_CHECK)
//  No altera la lógica de la búsqueda ni de los deltas: solo VERIFICA.
//  Maquinaria aislada aquí desde tabu.cpp (TAREA 2): lógica intacta.
// ===========================================================================
#include "tabu/solution.hpp"   // init_empty, bins_for_demand, compute_cost, ValidCandidate
#include <cmath>               // fabs, fmax
#include <cstdio>              // snprintf, fprintf
#include <cstdlib>             // abort
#include <limits>              // numeric_limits


// Comparación de doubles con tolerancia RELATIVA. Imprescindible aquí: rho
// puede llegar a ~1e11 (o más), y comparar cambios de coste con tolerancia
// ABSOLUTA daría falsos positivos por cancelación catastrófica. Un delta con
// un bug real difiere en cantidades discretas (un bin_cost ~400, una apertura
// ~4000, o una unidad de rho), que superan de sobra esta tolerancia relativa.
static bool oracle_close(double a, double b) {
  const double diff  = std::fabs(a - b);
  const double scale = std::fmax(std::fmax(std::fabs(a), std::fabs(b)), 1.0);
  return diff <= 1e-9 * scale + 1e-6;
}

// Reconstruye ESTADO + coste DESDE CERO a partir SOLO del conjunto abierto,
// SIN usar apply_open/apply_close (asignación nearest-open recalculada en un
// único barrido independiente). Es la "verdad terreno" del estado incremental.
static SolutionState oracle_rebuild(const SolutionState& cur,
                                    const Instance& instance, double rho) {
  SolutionState s;
  init_empty(s, instance);
  s.active = cur.active;   // mismas activaciones; todo lo demás se recalcula

  for (int i = 0; i < instance.n_buildings; ++i) {
    for (int k = 0; k < instance.n_waste_types; ++k) {
      int    chosen = -1;
      double cd     = std::numeric_limits<double>::infinity();
      // nearest-active: primer valid_candidate con el tipo k ACTIVO (lista ya ordenada).
      for (const ValidCandidate& vc : instance.valid_candidates[i][k]) {
        if (s.active[vc.j][k]) { chosen = vc.j; cd = vc.distance; break; }
      }
      s.assignment[i][k]    = chosen;
      s.assigned_dist[i][k] = cd;
      if (chosen != -1) {
        s.demand_at[chosen][k] += instance.demand[i][k];
        s.buildings_at[chosen][k].push_back(i);
      }
    }
  }
  for (int j = 0; j < instance.n_candidates; ++j) {
    for (int k = 0; k < instance.n_waste_types; ++k) {
      s.bins[j][k] = bins_for_demand(s.demand_at[j][k],
                                     instance.params.bin_capacity[k]);
    }
  }
  compute_cost(s, instance, rho);   // coste + violaciones desde el estado limpio
  return s;
}

// Verifica, tras aplicar un movimiento, que (A) el delta predicho coincide con
// el cambio real de coste, y (B) el estado incremental coincide con la
// reconstrucción desde cero. Si algo no cuadra, imprime diagnóstico y aborta.
void oracle_verify(const SolutionState& sol, const Instance& instance,
                   double rho, int iter, int move_type,
                   int p1, int p2, int p3,
                   double cost_before, double predicted_delta) {
  // Etiqueta del movimiento. Para el swap por tipo se inyecta k en la etiqueta,
  // y p1<->p2 = j_out<->j_in en los campos habituales.
  char mvbuf[24];
  const char* mv;
  if (move_type == 5) {
    std::snprintf(mvbuf, sizeof(mvbuf), "SWAPT k=%d", p3);
    mv = mvbuf;
  } else {
    mv = (move_type == 0 ? "CERRAR" :
          move_type == 1 ? "ABRIR " :
          move_type == 2 ? "SWAP  " :
          move_type == 3 ? "ACTIV " : "DESACT");
  }
  const bool  per_type = (move_type == 3 || move_type == 4);
  const bool  is_swap  = (move_type == 2 || move_type == 5);
  const char* sep = is_swap ? "<->" : per_type ? " k=" : "";
  const int   q2  = (is_swap || per_type) ? p2 : -1;
  bool ok = true;

  // --- (A) delta predicho vs cambio real de coste ---
  const double actual_delta = sol.total_cost - cost_before;
  if (!oracle_close(cost_before + predicted_delta, sol.total_cost)) {
    std::fprintf(stderr,
      "\n[ORACLE] FALLO (A) delta != cambio real | iter %d | mov %s %d%s%d\n"
      "  coste_antes             = %.10g\n"
      "  delta_predicho          = %.10g\n"
      "  delta_real              = %.10g\n"
      "  diferencia (pred-real)  = %.10g\n"
      "  coste_despues (real)    = %.10g\n"
      "  coste_despues (predicho)= %.10g\n",
      iter, mv, p1, sep, q2,
      cost_before, predicted_delta, actual_delta,
      predicted_delta - actual_delta, sol.total_cost,
      cost_before + predicted_delta);
    ok = false;
  }

  // --- (B) estado incremental vs reconstrucción desde cero ---
  const SolutionState fresh = oracle_rebuild(sol, instance, rho);

  if (sol.n_violations_capacity != fresh.n_violations_capacity ||
      sol.n_violations_coverage != fresh.n_violations_coverage) {
    std::fprintf(stderr,
      "\n[ORACLE] FALLO (B) violaciones incrementales != desde cero | iter %d | mov %s %d%s%d\n"
      "  incremental: cap=%d cov=%d\n"
      "  desde_cero : cap=%d cov=%d\n",
      iter, mv, p1, sep, q2,
      sol.n_violations_capacity, sol.n_violations_coverage,
      fresh.n_violations_capacity, fresh.n_violations_coverage);
    ok = false;
  }
  if (!oracle_close(sol.total_cost, fresh.total_cost)) {
    std::fprintf(stderr,
      "\n[ORACLE] FALLO (B) total_cost incremental != desde cero | iter %d | mov %s %d%s%d\n"
      "  total_cost incremental = %.10g\n"
      "  total_cost desde_cero  = %.10g\n"
      "  diferencia             = %.10g\n",
      iter, mv, p1, sep, q2,
      sol.total_cost, fresh.total_cost, sol.total_cost - fresh.total_cost);
    ok = false;
  }
  // Bins por punto/tipo: comparación exacta (int). Ante un fallo, se muestra
  // también la demanda para distinguir un bug real de un empate en frontera ceil.
  for (int j = 0; j < instance.n_candidates; ++j) {
    for (int k = 0; k < instance.n_waste_types; ++k) {
      if (sol.bins[j][k] != fresh.bins[j][k]) {
        std::fprintf(stderr,
          "\n[ORACLE] FALLO (B) bins incrementales != desde cero | iter %d | mov %s %d%s%d\n"
          "  punto %d tipo %d: bins incremental=%d  desde_cero=%d\n"
          "  demanda incremental=%.10g  desde_cero=%.10g  (dif=%.3g)\n",
          iter, mv, p1, sep, q2, j, k,
          sol.bins[j][k], fresh.bins[j][k],
          sol.demand_at[j][k], fresh.demand_at[j][k],
          sol.demand_at[j][k] - fresh.demand_at[j][k]);
        ok = false;
        j = instance.n_candidates;   // cortar ambos bucles: un fallo basta
        break;
      }
    }
  }

  if (!ok) {
    std::fprintf(stderr,
      "[ORACLE] Inconsistencia detectada: abortando en iter %d.\n", iter);
    std::fflush(stderr);
    std::abort();
  }
}
#endif  // ORACLE_CHECK
