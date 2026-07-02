#include "tabu/tabu.hpp"
#include "tabu/movimientos.hpp"
#include <limits>
#include <set>
#include <chrono>

#ifdef ORACLE_CHECK
// El oráculo de consistencia solo compila cuando ORACLE_CHECK está definido.
// Con el flag desactivado, este bloque desaparece y el comportamiento es idéntico.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#endif



/** Inicializa para n_candidates puntos y n_types tipos, sin ningún veto activo. */
void TabuList::init(int n_candidates, int n_types) {
  forbid_open_until.assign(n_candidates, -1);
  forbid_close_until.assign(n_candidates, -1);
  forbid_activate_until.assign(n_candidates, std::vector<int>(n_types, -1));
  forbid_deactivate_until.assign(n_candidates, std::vector<int>(n_types, -1));
}

/** ¿Está vetado ABRIR el punto j en la iteración actual? */
bool TabuList::is_open_forbidden(int j, int current_iter) const {
  return forbid_open_until[j] >= current_iter;
}

/** ¿Está vetado CERRAR el punto j en la iteración actual? */
bool TabuList::is_close_forbidden(int j, int current_iter) const {
  return forbid_close_until[j] >= current_iter;
}

/** Tras CERRAR j: prohibir REABRIRLO durante `tenure` iteraciones. */
void TabuList::mark_closed(int j, int current_iter, int tenure) {
  forbid_open_until[j] = current_iter + tenure;
}

/** Tras ABRIR j: prohibir CERRARLO durante `tenure` iteraciones. */
void TabuList::mark_opened(int j, int current_iter, int tenure) {
  forbid_close_until[j] = current_iter + tenure;
}

/** ¿Está vetado ACTIVAR el tipo k en el punto j? */
bool TabuList::is_activate_forbidden(int j, int k, int current_iter) const {
  return forbid_activate_until[j][k] >= current_iter;
}

/** ¿Está vetado DESACTIVAR el tipo k en el punto j? */
bool TabuList::is_deactivate_forbidden(int j, int k, int current_iter) const {
  return forbid_deactivate_until[j][k] >= current_iter;
}

/** Tras ACTIVAR (j,k): prohibir DESACTIVAR ese tipo en ese punto un tenure. */
void TabuList::mark_activated(int j, int k, int current_iter, int tenure) {
  forbid_deactivate_until[j][k] = current_iter + tenure;
}

/** Tras DESACTIVAR (j,k): prohibir REACTIVAR ese tipo en ese punto un tenure. */
void TabuList::mark_deactivated(int j, int k, int current_iter, int tenure) {
  forbid_activate_until[j][k] = current_iter + tenure;
}


// ¿Es la solución factible? (sin violaciones de ningún tipo)
static bool is_feasible(const SolutionState& s) {
  return s.n_violations_capacity == 0 && s.n_violations_coverage == 0;
}


#ifdef ORACLE_CHECK
// ===========================================================================
//  ORÁCULO DE CONSISTENCIA (solo con -DORACLE_CHECK)
//  No altera la lógica de la búsqueda ni de los deltas: solo VERIFICA.
// ===========================================================================

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
static void oracle_verify(const SolutionState& sol, const Instance& instance,
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


SolutionState tabu_search(SolutionState solution, const Instance& instance,
                          const TabuParams& params) {
  const int n_candidates = instance.n_candidates;
  double current_rho = params.rho;          // rho evoluciona durante la búsqueda
  const double rho_base = params.rho;       // valor al que resetear
  const double rho_factor = 1.5;            // cuánto sube cada vez
  const int rho_patience = 50;              // iteraciones sin factible antes de subir
  int iters_since_feasible = 0;

  // Coste de partida.
  compute_cost(solution, instance, current_rho);

  // Récords: mejor global (para aspiración) y mejor factible (lo que devolvemos).
  SolutionState best_global   = solution;
  SolutionState best_feasible = solution;
  bool have_feasible = is_feasible(solution);

  TabuList tabu;
  tabu.init(n_candidates, instance.n_waste_types);

  const int report_every = 100;

  // Cronómetro: tiempo transcurrido desde el inicio de la búsqueda.
  const auto t_start = std::chrono::steady_clock::now();

  for (int iter = 0; iter < params.max_iters; ++iter) {
    // Mejor movimiento permitido de esta iteración.
    double best_delta = std::numeric_limits<double>::infinity();
    int best_point   = -1;   // punto principal (abre/cierra/activa/desactiva; swap: j_out)
    int best_second  = -1;   // per-tipo: el tipo k; swap: j_in
    int best_third   = -1;   // swap por tipo: el tipo k
    int best_move_type = -1; // 0=cerrar 1=abrir 3=activar(j,k) 4=desactivar(j,k) 5=swap(j_out,j_in,k)

    // --- Vecinos ENTEROS: abrir / cerrar el punto completo ---
    //     Se conservan junto a los per-tipo: cerrar un punto de 4 tipos en un solo
    //     paso reembolsa C_j de golpe (vía per-tipo solo lo haría la última
    //     desactivación, y las intermedias no serían mejorantes → punto atascado).
    for (int j = 0; j < n_candidates; ++j) {
      if (solution.is_open(j)) {
        // CERRAR j (entero)
        double d = delta_close(solution, instance, j, current_rho);
        bool tabu_move = tabu.is_close_forbidden(j, iter);
        bool aspires = (solution.total_cost + d < best_global.total_cost);
        if ((!tabu_move || aspires) && d < best_delta) {
          best_delta = d; best_point = j; best_second = -1; best_move_type = 0;
        }
      } else {
        // ABRIR j (entero)
        double d = delta_open(solution, instance, j, current_rho);
        bool tabu_move = tabu.is_open_forbidden(j, iter);
        bool aspires = (solution.total_cost + d < best_global.total_cost);
        if ((!tabu_move || aspires) && d < best_delta) {
          best_delta = d; best_point = j; best_second = -1; best_move_type = 1;
        }
      }
    }

    // --- Vecinos POR TIPO: activar / desactivar un solo (j,k) ---
    //     Esto ROMPE el régimen colapsado: un punto puede quedar con unos tipos
    //     activos y otros no. El veto tabú aquí es por (j,k), no por punto.
    for (int j = 0; j < n_candidates; ++j) {
      for (int k = 0; k < instance.n_waste_types; ++k) {
        if (solution.active[j][k]) {
          // DESACTIVAR (j,k)
          double d = delta_deactivate(solution, instance, j, k, current_rho);
          bool tabu_move = tabu.is_deactivate_forbidden(j, k, iter);
          bool aspires = (solution.total_cost + d < best_global.total_cost);
          if ((!tabu_move || aspires) && d < best_delta) {
            best_delta = d; best_point = j; best_second = k; best_move_type = 4;
          }
        } else {
          // ACTIVAR (j,k)  (j puede estar cerrado, o abierto con otros tipos)
          double d = delta_activate(solution, instance, j, k, current_rho);
          bool tabu_move = tabu.is_activate_forbidden(j, k, iter);
          bool aspires = (solution.total_cost + d < best_global.total_cost);
          if ((!tabu_move || aspires) && d < best_delta) {
            best_delta = d; best_point = j; best_second = k; best_move_type = 3;
          }
        }
      }
    }

    // --- Vecinos SWAP POR TIPO, ACOTADOS por cercanía: mover el tipo k de un
    //     punto activo j_out a un j_in cercano (que no sirva k). Los j_in candidatos
    //     son la UNIÓN de valid_candidates de los edificios de (j_out,k) — el conjunto
    //     de puntos que pueden servir a alguno de ellos dentro de su radio r_k. ---
    for (int j_out = 0; j_out < n_candidates; ++j_out) {
      for (int k = 0; k < instance.n_waste_types; ++k) {
        if (!solution.active[j_out][k]) continue;   // solo tipos que j_out sirve

        // Reunir los j_in cercanos que NO sirven ya el tipo k (dedup con set).
        std::set<int> nearby;
        for (int i : solution.buildings_at[j_out][k]) {
          for (const ValidCandidate& vc : instance.valid_candidates[i][k]) {
            if (vc.j != j_out && !solution.active[vc.j][k]) nearby.insert(vc.j);
          }
        }

        for (int j_in : nearby) {
          double d = delta_swap_type(solution, instance, j_out, j_in, k, current_rho);
          // Tabú del swap: veta si cualquiera de sus dos mitades está vetada.
          bool tabu_move = tabu.is_deactivate_forbidden(j_out, k, iter) ||
                           tabu.is_activate_forbidden(j_in, k, iter);
          bool aspires = (solution.total_cost + d < best_global.total_cost);
          if ((!tabu_move || aspires) && d < best_delta) {
            best_delta = d; best_point = j_out; best_second = j_in;
            best_third = k; best_move_type = 5;
          }
        }
      }
    }

    // Sin movimiento permitido: terminar.
    if (best_move_type == -1) break;

#ifdef ORACLE_CHECK
    // Foto del coste ANTES de aplicar y del delta que la búsqueda predice.
    // (En este punto solution.total_cost es la base que asumieron los deltas.)
    const double oracle_cost_before     = solution.total_cost;
    const double oracle_predicted_delta = best_delta;
#endif

    // --- Aplicar el ganador con la primitiva, y marcar tabú ---
    if (best_move_type == 0) {
      // Cerrar best_point (entero) → vetar REABRIRLO.
      apply_close(solution, instance, best_point);
      tabu.mark_closed(best_point, iter, params.tabu_tenure);
    } else if (best_move_type == 1) {
      // Abrir best_point (entero) → vetar CERRARLO.
      apply_open(solution, instance, best_point);
      tabu.mark_opened(best_point, iter, params.tabu_tenure);
    } else if (best_move_type == 3) {
      // Activar (j,k) → vetar DESACTIVAR ese tipo en ese punto.
      apply_activate(solution, instance, best_point, best_second);
      tabu.mark_activated(best_point, best_second, iter, params.tabu_tenure);
    } else if (best_move_type == 4) {
      // Desactivar (j,k) → vetar REACTIVAR ese tipo en ese punto.
      apply_deactivate(solution, instance, best_point, best_second);
      tabu.mark_deactivated(best_point, best_second, iter, params.tabu_tenure);
    } else {  // best_move_type == 5: swap por tipo (j_out=best_point, j_in=best_second, k=best_third)
      // Mover k de j_out a j_in → vetar traerlo de vuelta (reactivar en j_out) y
      // vetar quitarlo de j_in (desactivar allí): impide el swap inverso inmediato.
      apply_swap_type(solution, instance, best_point, best_second, best_third);
      tabu.mark_deactivated(best_point,  best_third, iter, params.tabu_tenure);  // veta reactivar (j_out,k)
      tabu.mark_activated(best_second, best_third, iter, params.tabu_tenure);    // veta desactivar (j_in,k)
    }

    // Recalcular el coste de la nueva solución actual.
    compute_cost(solution, instance, current_rho);

#ifdef ORACLE_CHECK
    // Verificar delta y estado con el mismo rho usado en el delta y este cómputo.
    oracle_verify(solution, instance, current_rho, iter, best_move_type,
                  best_point, best_second, best_third,
                  oracle_cost_before, oracle_predicted_delta);
#endif

    // Actualizar récord global (para aspiración).
    if (solution.total_cost < best_global.total_cost) {
      best_global = solution;
    }

    // Actualizar mejor factible (lo que devolveremos).
    if (is_feasible(solution) &&
        (!have_feasible || solution.total_cost < best_feasible.total_cost)) {
      best_feasible = solution;
      have_feasible = true;
    }

    // --- Penalización rho ADAPTATIVA ---
    // Si la solución actual es factible, la penalización no ha hecho falta:
    // reseteamos rho a su valor base (y recalculamos coste, que depende de rho).
    // Si llevamos `rho_patience` iteraciones sin pisar factibilidad, subimos rho
    // para empujar la búsqueda hacia soluciones sin violaciones.
    if (is_feasible(solution)) {
      iters_since_feasible = 0;
      if (current_rho != rho_base) {
        current_rho = rho_base;
        compute_cost(solution, instance, current_rho);  // recalcular con rho reseteado
      }
    } else {
      iters_since_feasible++;
      if (iters_since_feasible >= rho_patience) {
        current_rho = std::min(current_rho * rho_factor, rho_base * 1e6);
        iters_since_feasible = 0;
        compute_cost(solution, instance, current_rho);  // recalcular con rho subido
      }
    }

    // --- Heartbeat ---
    if (iter % report_every == 0) {
      const double elapsed_s =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - t_start).count();
      const char* move = (best_move_type == 0 ? "CERRAR" :
                          best_move_type == 1 ? "ABRIR " :
                          best_move_type == 3 ? "ACTIV " :
                          best_move_type == 4 ? "DESACT" : "SWAPT ");
      std::cout << "  iter " << iter << " | " << move << " " << best_point;
      if (best_move_type == 3 || best_move_type == 4) std::cout << " k=" << best_second;
      if (best_move_type == 5) std::cout << "<->" << best_second << " k=" << best_third;
      std::cout << " | delta: " << best_delta
                << " | coste: " << solution.total_cost
                << " | mejor fact: " << (have_feasible ? best_feasible.total_cost : -1)
                << " | rho: " << current_rho
                << " | t: " << elapsed_s << "s"
                << "\n";
    }
  }

  // Devolver la mejor factible (o el mejor global si nunca se halló factible).
  return have_feasible ? best_feasible : best_global;
}