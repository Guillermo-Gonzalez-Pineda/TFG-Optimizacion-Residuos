#pragma once

#include <vector>
#include <iostream>
#include "tabu/solution.hpp"
#include "tabu/instancia.hpp"

/** Parámetros de la búsqueda tabú. Único lugar donde vive rho. */
struct TabuParams {
  double rho = 100000.0;
  int tabu_tenure = 20;
  int max_iters = 10000;
};

/**
 * Lista tabú con DOS granularidades:
 *  - Por PUNTO (forbid_open/close_until): para los movimientos ENTEROS abrir/cerrar.
 *  - Por (j,k) (forbid_activate/deactivate_until): para los movimientos POR TIPO.
 * Cada familia de movimientos usa su propia granularidad natural. El tabú solo
 * afecta a la SELECCIÓN de movimiento, nunca a la contabilidad de coste.
 */
struct TabuList {
  std::vector<int> forbid_open_until;                    // [j]  (movimientos enteros)
  std::vector<int> forbid_close_until;                   // [j]  (movimientos enteros)
  std::vector<std::vector<int>> forbid_activate_until;   // [j][k] (movimientos por tipo)
  std::vector<std::vector<int>> forbid_deactivate_until; // [j][k] (movimientos por tipo)

  void init(int n_candidates, int n_types);

  bool is_open_forbidden(int j, int it) const;
  bool is_close_forbidden(int j, int it) const;
  void mark_closed(int j, int it, int tenure);
  void mark_opened(int j, int it, int tenure);

  bool is_activate_forbidden(int j, int k, int it) const;
  bool is_deactivate_forbidden(int j, int k, int it) const;
  void mark_activated(int j, int k, int it, int tenure);    // veta DESACTIVAR (j,k)
  void mark_deactivated(int j, int k, int it, int tenure);  // veta REACTIVAR (j,k)
};

/**
 * Ejecuta la búsqueda tabú partiendo de `solution` (ya construida). Mejora la
 * solución mediante movimientos de abrir/cerrar, y devuelve la mejor solución
 * FACTIBLE encontrada.
 *
 * @param solution Solución inicial (se usa como punto de partida).
 * @param instance Instancia del problema.
 * @param params Parámetros de la búsqueda.
 * @return La mejor solución factible encontrada durante la búsqueda.
 */
SolutionState tabu_search(SolutionState solution, const Instance& instance,
                          const TabuParams& params);