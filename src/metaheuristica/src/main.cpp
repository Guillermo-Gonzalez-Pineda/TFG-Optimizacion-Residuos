#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include "tabu/instancia.hpp"
#include "tabu/solution.hpp"
#include "tabu/constructive.hpp"
#include "tabu/io.hpp"
#include "tabu/movimientos.hpp"
#include "tabu/tabu.hpp"

#ifdef PRIMITIVE_TEST
// ===========================================================================
//  TEST DE PRIMITIVAS PER-TIPO (solo con -DPRIMITIVE_TEST)
//  Ejercita apply_activate / apply_deactivate EN AISLAMIENTO: el oráculo normal
//  solo verifica lo que la búsqueda llama (apply_open/close/swap), y estas
//  primitivas aún no están enganchadas. Devuelve nº de fallos como exit code.
// ===========================================================================
#include <cmath>

static int g_fails = 0;

static void check(bool cond, const std::string& msg) {
  std::cout << (cond ? "  [ok]   " : "  [FAIL] ") << msg << "\n";
  if (!cond) ++g_fails;
}

// Igualdad relativa laxa para costes/demanda (evita falsos negativos por FP).
static bool close_rel(double a, double b) {
  const double scale = std::fmax(std::fmax(std::fabs(a), std::fabs(b)), 1.0);
  return std::fabs(a - b) <= 1e-9 * scale + 1e-6;
}

// Copia ordenada: comparamos buildings_at[j][k] como CONJUNTO (el orden de
// inserción puede diferir entre apply_open y 4×apply_activate; el conjunto no).
static std::vector<int> sorted_copy(std::vector<int> v) {
  std::sort(v.begin(), v.end());
  return v;
}

// Compara dos soluciones en todos los campos que definen el estado + coste.
static bool states_equal(const SolutionState& a, const SolutionState& b,
                         const Instance& inst) {
  const int J = inst.n_candidates, I = inst.n_buildings, K = inst.n_waste_types;
  for (int j = 0; j < J; ++j)
    for (int k = 0; k < K; ++k) {
      if (a.active[j][k] != b.active[j][k]) return false;
      if (a.bins[j][k]   != b.bins[j][k])   return false;
      if (!close_rel(a.demand_at[j][k], b.demand_at[j][k])) return false;
      if (sorted_copy(a.buildings_at[j][k]) != sorted_copy(b.buildings_at[j][k])) return false;
    }
  for (int i = 0; i < I; ++i)
    for (int k = 0; k < K; ++k)
      if (a.assignment[i][k] != b.assignment[i][k]) return false;
  if (a.n_violations_capacity != b.n_violations_capacity) return false;
  if (a.n_violations_coverage != b.n_violations_coverage) return false;
  return close_rel(a.total_cost, b.total_cost);
}

static int run_primitive_tests(const Instance& instance) {
  const int K = instance.n_waste_types;
  const double rho = 100000.0;
  std::cout << "=== TEST DE PRIMITIVAS PER-TIPO (apply_activate / apply_deactivate) ===\n";

  // Punto de prueba: el candidato con más edificios (para ejercitar migraciones).
  int p = 0; size_t best = 0;
  for (int j = 0; j < instance.n_candidates; ++j)
    if (instance.buildings_of[j].size() > best) { best = instance.buildings_of[j].size(); p = j; }
  const double Cp = instance.candidates[p].opening_cost;
  std::cout << "punto p=" << p << " | C_p=" << Cp
            << " | |buildings_of[p]|=" << instance.buildings_of[p].size() << "\n";

  // --- [A] activar los 4 tipos uno a uno → C_j cobrado EXACTAMENTE una vez ---
  std::cout << "\n[A] activar 4 tipos de un punto cerrado\n";
  {
    SolutionState s; init_empty(s, instance);            // total_cost = 0
    for (int k = 0; k < K; ++k) {
      apply_activate(s, instance, p, k);
      check(close_rel(s.total_cost, Cp),
            "tras activar tipo " + std::to_string(k) + ": total_cost == C_p");
    }
    check(s.is_open(p), "el punto quedó abierto");
    check(close_rel(s.total_cost, Cp), "C_j sumado exactamente UNA vez (no 4)");
  }

  // --- [B] desactivar los 4 → C_j reembolsado EXACTAMENTE una vez (la última) ---
  std::cout << "\n[B] desactivar 4 tipos\n";
  {
    SolutionState s; init_empty(s, instance);
    for (int k = 0; k < K; ++k) apply_activate(s, instance, p, k);   // total_cost = Cp
    for (int k = 0; k < K; ++k) {
      apply_deactivate(s, instance, p, k);
      const bool last = (k == K - 1);
      check(close_rel(s.total_cost, last ? 0.0 : Cp),
            "tras desactivar tipo " + std::to_string(k) +
            (last ? ": total_cost vuelve a 0 (reembolso)" : ": total_cost sigue C_p"));
    }
    check(!s.is_open(p), "el punto quedó cerrado");
    bool vacio = true;
    for (int k = 0; k < K; ++k)
      if (s.demand_at[p][k] != 0.0 || s.bins[p][k] != 0 || !s.buildings_at[p][k].empty())
        vacio = false;
    check(vacio, "punto totalmente vacío tras cerrar");
  }

  // --- [C] activar los 4 == apply_open entero (mismo estado + coste) ---
  std::cout << "\n[C] 4×apply_activate equivale a apply_open entero\n";
  {
    SolutionState base; construct_initial(base, instance, rho);   // estado realista
    int q = -1;
    for (int j = 0; j < instance.n_candidates; ++j)
      if (!base.is_open(j) && !instance.buildings_of[j].empty()) { q = j; break; }
    if (q == -1) {
      check(false, "no hay candidato cerrado con edificios (test no aplicable)");
    } else {
      std::cout << "  abriendo candidato cerrado q=" << q << "\n";
      SolutionState s_open = base;
      apply_open(s_open, instance, q);
      compute_cost(s_open, instance, rho);

      SolutionState s_act = base;
      for (int k = 0; k < K; ++k) apply_activate(s_act, instance, q, k);
      compute_cost(s_act, instance, rho);

      check(states_equal(s_open, s_act, instance),
            "estado y coste idénticos (apply_open vs 4×apply_activate)");
    }
  }

  // --- [D] punto a medias (solo tipos 0 y 2) coherente ---
  std::cout << "\n[D] punto a medias: solo tipos 0 y 2 activos\n";
  {
    SolutionState s; init_empty(s, instance);
    apply_activate(s, instance, p, 0);
    apply_activate(s, instance, p, 2);
    check(s.is_open(p), "is_open(p) == true con solo 2 tipos");
    check(s.active[p][0] && s.active[p][2], "tipos 0 y 2 activos");
    check(!s.active[p][1] && !s.active[p][3], "tipos 1 y 3 inactivos");
    check(s.demand_at[p][1] == 0.0 && s.demand_at[p][3] == 0.0, "tipos inactivos sin demanda");
    check(s.bins[p][1] == 0 && s.bins[p][3] == 0, "tipos inactivos sin bins");
    check(s.buildings_at[p][1].empty() && s.buildings_at[p][3].empty(),
          "tipos inactivos sin edificios");
    bool ninguno_13 = true;
    for (int i = 0; i < instance.n_buildings; ++i)
      if (s.assignment[i][1] == p || s.assignment[i][3] == p) ninguno_13 = false;
    check(ninguno_13, "ningún residuo de tipo 1/3 asignado a p");
    check(close_rel(s.total_cost, Cp), "C_j cobrado una vez (2 tipos, no 2×C_j)");
  }

  std::cout << "\n=== RESULTADO: "
            << (g_fails == 0 ? "TODO OK" : std::to_string(g_fails) + " FALLOS")
            << " ===\n";
  return g_fails;
}
#endif  // PRIMITIVE_TEST

int main(int argc, char** argv) {
  // Ruta a la instancia de 500m (se puede sobrescribir por argumento).
  const std::string path = (argc > 1)
      ? argv[1]
      : "../../data/processed/instancia_laguna_500m.json";

  // Cronometramos el cómputo (carga + preproceso + construcción).
  const auto t0 = std::chrono::steady_clock::now();

  Instance instance = load_instance(path);
  preprocess(instance);

#ifdef PRIMITIVE_TEST
  // Con -DPRIMITIVE_TEST el binario es un runner de tests de primitivas: no corre tabú.
  return run_primitive_tests(instance);
#endif

  // Test apply_close: abrir y cerrar debe dejar el coste como al principio.
  SolutionState solution;

  construct_initial(solution, instance, 100000.0);
  double cost_greedy = solution.total_cost;

  TabuParams params;   // valores por defecto
  SolutionState result = tabu_search(solution, instance, params);

  std::cout << "\n=== Tabu search ===\n";
  std::cout << "  Coste greedy:  " << cost_greedy << "\n";
  std::cout << "  Coste tabu:    " << result.total_cost << "\n";
  std::cout << "  Mejora:        " << (cost_greedy - result.total_cost)
            << " (" << 100.0*(cost_greedy - result.total_cost)/cost_greedy << "%)\n";
  std::cout << "  Factible: " << (result.n_violations_capacity==0 &&
                                  result.n_violations_coverage==0 ? "SI" : "NO") << "\n";

  return 0;
}