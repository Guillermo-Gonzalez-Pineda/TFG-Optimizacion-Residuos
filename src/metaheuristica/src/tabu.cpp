#include "tabu/tabu.hpp"
#include "tabu/movimientos.hpp"
#include <limits>
#include <set>
#include <chrono>



/** Inicializa para n_candidates puntos, sin ningún veto activo. */
void TabuList::init(int n_candidates) {
  forbid_open_until.assign(n_candidates, -1);
  forbid_close_until.assign(n_candidates, -1);
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


// ¿Es la solución factible? (sin violaciones de ningún tipo)
static bool is_feasible(const SolutionState& s) {
  return s.n_violations_capacity == 0 && s.n_violations_coverage == 0;
}


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
  tabu.init(n_candidates);

  const int report_every = 100;

  // Cronómetro: tiempo transcurrido desde el inicio de la búsqueda.
  const auto t_start = std::chrono::steady_clock::now();

  for (int iter = 0; iter < params.max_iters; ++iter) {
    // Mejor movimiento permitido de esta iteración.
    double best_delta = std::numeric_limits<double>::infinity();
    int best_point   = -1;   // punto principal (el que se cierra/abre; en swap, j_out)
    int best_second  = -1;   // segundo punto (solo swap: j_in)
    int best_move_type = -1; // 0 = cerrar, 1 = abrir, 2 = swap

    // --- Vecinos de abrir / cerrar ---
    for (int j = 0; j < n_candidates; ++j) {
      if (solution.open[j]) {
        // CERRAR j
        double d = delta_close(solution, instance, j, current_rho);
        bool tabu_move = tabu.is_close_forbidden(j, iter);
        bool aspires = (solution.total_cost + d < best_global.total_cost);
        if ((!tabu_move || aspires) && d < best_delta) {
          best_delta = d; best_point = j; best_second = -1; best_move_type = 0;
        }
      } else {
        // ABRIR j
        double d = delta_open(solution, instance, j, current_rho);
        bool tabu_move = tabu.is_open_forbidden(j, iter);
        bool aspires = (solution.total_cost + d < best_global.total_cost);
        if ((!tabu_move || aspires) && d < best_delta) {
          best_delta = d; best_point = j; best_second = -1; best_move_type = 1;
        }
      }
    }

    // --- Vecinos de swap ACOTADOS: por cada abierto, solo intercambiar por
    //     candidatos cerrados CERCANOS (los valid_candidates de sus edificios). ---
    for (int j_out = 0; j_out < n_candidates; ++j_out) {
      if (!solution.open[j_out]) continue;

      // Recoger candidatos cerrados cercanos: los valid_candidates de los
      // edificios que j_out sirve. Usamos un set para no repetir.
      std::set<int> nearby_closed;
      for (const auto& [i, k] : solution.buildings_at[j_out]) {
        for (const ValidCandidate& vc : instance.valid_candidates[i][k]) {
          if (!solution.open[vc.j] && vc.j != j_out) {
            nearby_closed.insert(vc.j);
          }
        }
      }

      for (int j_in : nearby_closed) {
        double d = delta_swap(solution, instance, j_out, j_in, current_rho);
        bool tabu_move = tabu.is_close_forbidden(j_out, iter) ||
                         tabu.is_open_forbidden(j_in, iter);
        bool aspires = (solution.total_cost + d < best_global.total_cost);
        if ((!tabu_move || aspires) && d < best_delta) {
          best_delta = d; best_point = j_out; best_second = j_in; best_move_type = 2;
        }
      }
    }

    // Sin movimiento permitido: terminar.
    if (best_move_type == -1) break;

    // --- Aplicar el ganador con la primitiva, y marcar tabú ---
    if (best_move_type == 0) {
      // Cerrar best_point → vetar REABRIRLO.
      apply_close(solution, instance, best_point);
      tabu.mark_closed(best_point, iter, params.tabu_tenure);
    } else if (best_move_type == 1) {
      // Abrir best_point → vetar CERRARLO.
      apply_open(solution, instance, best_point);
      tabu.mark_opened(best_point, iter, params.tabu_tenure);
    } else {
      // Swap: best_point = j_out (se cierra), best_second = j_in (se abre).
      // REVISAR (mañana): confirmar que el veto va al punto correcto.
      //   - j_out se cierra → mark_closed(j_out) veta reabrirlo.
      //   - j_in  se abre   → mark_opened(j_in)  veta cerrarlo.
      apply_swap(solution, instance, best_point, best_second);
      tabu.mark_closed(best_point,  iter, params.tabu_tenure);
      tabu.mark_opened(best_second, iter, params.tabu_tenure);
    }

    // Recalcular el coste de la nueva solución actual.
    compute_cost(solution, instance, current_rho);

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
                          best_move_type == 1 ? "ABRIR " : "SWAP  ");
      std::cout << "  iter " << iter << " | " << move << " " << best_point;
      if (best_move_type == 2) std::cout << "<->" << best_second;
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