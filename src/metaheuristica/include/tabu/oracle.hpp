#ifndef TABU_ORACLE_HPP
#define TABU_ORACLE_HPP

// ===========================================================================
//  ORÁCULO DE CONSISTENCIA — interfaz
//
//  Juez que, tras cada movimiento aplicado, verifica dos invariantes:
//    (A) el delta predicho por la búsqueda coincide con el cambio real de coste
//        (cost_before + delta ≈ total_cost), y
//    (B) el estado incremental (bins, violaciones, coste) coincide con una
//        reconstrucción desde cero por nearest-active.
//  Comparaciones con tolerancia RELATIVA; ante cualquier discrepancia imprime
//  diagnóstico por stderr y aborta. NO altera la lógica de la búsqueda.
//
//  Todo el oráculo vive tras -DORACLE_CHECK. En producción (flag apagado) este
//  header no declara ni incluye nada: las dos macros de abajo se expanden a un
//  no-op y NO evalúan sus argumentos, de modo que el binario no contiene ningún
//  símbolo, variable ni evaluación del oráculo (coste cero garantizado por el
//  preprocesador, no por el optimizador).
// ===========================================================================

#ifdef ORACLE_CHECK
#include "tabu/solution.hpp"   // SolutionState, Instance (vía instancia.hpp)

/**
 * Verifica los invariantes (A) y (B) tras aplicar un movimiento. Si fallan,
 * imprime diagnóstico y aborta (std::abort). No modifica `sol`.
 *
 * @param sol             solución YA con el movimiento aplicado y coste recalculado.
 * @param instance        instancia del problema.
 * @param rho             penalización usada en el delta y en el compute_cost actual.
 * @param iter            iteración actual (para el diagnóstico).
 * @param move_type       0=cerrar 1=abrir 2=swap 3=activar 4=desactivar 5=swap-tipo.
 * @param p1,p2,p3        operandos del movimiento (punto, tipo/segundo punto, tipo).
 * @param cost_before     coste total ANTES de aplicar el movimiento.
 * @param predicted_delta delta que la búsqueda predijo para el movimiento.
 */
void oracle_verify(const SolutionState& sol, const Instance& instance,
                   double rho, int iter, int move_type,
                   int p1, int p2, int p3,
                   double cost_before, double predicted_delta);

/**
 * Verifica el INVARIANTE DE COLAPSO del modo entero: si `active` (modo entero),
 * todo punto j debe tener sus tipos uniformes (todos activos = abierto entero, o
 * todos inactivos = cerrado). Un punto "a medias" significa que la búsqueda se
 * salió del régimen colapsado → aborta con diagnóstico. En per-tipo (`active`
 * false) no hace nada: los puntos a medias son legítimos.
 */
void oracle_assert_integer_regime(const SolutionState& sol, const Instance& instance,
                                  int iter, bool active);

// Captura los datos SOLO-para-oráculo (coste previo y delta predicho) ANTES de
// aplicar el movimiento. Declara las variables únicamente en modo debug.
#define ORACLE_SNAPSHOT(cost_before, predicted_delta, cost_expr, delta_expr) \
    const double cost_before     = (cost_expr);                              \
    const double predicted_delta = (delta_expr)

// Punto de verificación: una sola línea en el bucle de la búsqueda.
#define ORACLE_VERIFY(...) oracle_verify(__VA_ARGS__)

// Invariante de régimen (modo entero): una sola línea en el bucle.
#define ORACLE_ASSERT_INTEGER(...) oracle_assert_integer_regime(__VA_ARGS__)

#else  // !ORACLE_CHECK  → no-ops: los argumentos NO se evalúan ni existen.

#define ORACLE_SNAPSHOT(cost_before, predicted_delta, cost_expr, delta_expr) ((void)0)
#define ORACLE_VERIFY(...)                                                   ((void)0)
#define ORACLE_ASSERT_INTEGER(...)                                           ((void)0)

#endif  // ORACLE_CHECK

#endif  // TABU_ORACLE_HPP
