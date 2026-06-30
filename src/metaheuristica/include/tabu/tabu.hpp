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

/** (TabuList struct — el que ya validaste — va aquí también.) */
struct TabuList {
  std::vector<int> forbid_open_until;
  std::vector<int> forbid_close_until;
  void init(int n);
  bool is_open_forbidden(int j, int it) const;
  bool is_close_forbidden(int j, int it) const;
  void mark_closed(int j, int it, int tenure);
  void mark_opened(int j, int it, int tenure);
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