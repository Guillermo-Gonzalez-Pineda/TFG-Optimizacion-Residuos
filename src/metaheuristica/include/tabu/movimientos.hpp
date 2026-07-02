#pragma once

#include "tabu/solution.hpp"
#include "tabu/instancia.hpp"

/**
 * Calcula el CAMBIO de coste que produciría cerrar el punto `candidate`, SIN
 * modificar la solución. Devuelve delta = coste(tras cerrar) - coste(actual).
 * Un delta negativo significa que cerrar mejora (abarata).
 *
 * Simula el efecto de forma ligera: lee el estado, acumula en variables locales
 * el cambio en apertura, contenedores y violaciones de los puntos afectados, y
 * devuelve el número. No copia ni muta el estado.
 *
 * @param solution Solución actual (solo se lee).
 * @param instance Instancia (geometría, derivadas, costes).
 * @param candidate Punto cuyo cierre se evalúa.
 * @param rho Peso de la penalización por violación.
 * @return Cambio de coste (negativo = mejora).
 */
double delta_close(const SolutionState& solution, const Instance& instance,
                   int candidate, double rho);


/**
 * Calcula el CAMBIO de coste que produciría abrir el punto `candidate`, SIN
 * modificar la solución. Devuelve delta = coste(tras abrir) - coste(actual).
 * Un delta negativo significa que abrir mejora.
 *
 * Simula de forma ligera: los edificios para los que 'candidate' es más cercano
 * que su asignación actual migrarían a él; se acumula la demanda que gana
 * 'candidate' y la que pierden los puntos antiguos, y se calcula el cambio de
 * contenedores y violaciones. No copia ni muta el estado.
 */
double delta_open(const SolutionState& solution, const Instance& instance,
                  int candidate, double rho);



/**
 * Calcula el cambio de coste de intercambiar (cerrar j_out, abrir j_in) sin
 * tocar la solución original. Por ahora se evalúa sobre una copia temporal,
 * que garantiza correctitud frente a la interacción cierre/apertura (los
 * huérfanos de j_out pueden reubicarse en j_in y viceversa). Optimizable más
 * adelante a evaluación puramente local si hiciera falta velocidad.
 */
double delta_swap(const SolutionState& solution, const Instance& instance,
                  int j_out, int j_in, double rho);


/**
 * Calcula el CAMBIO de coste de ACTIVAR el tipo `k` en el punto `j`, SIN mutar el
 * estado. Es la cara "simulación" de apply_activate. Predice el término de
 * apertura z[j]: suma C_j SOLO si el punto está CERRADO ahora (!is_open(j)), pues
 * solo entonces activar abre el punto. El resto: bins del tipo k que j gana por la
 * demanda que atrae (partiendo de su demanda actual de k), y el ahorro en los
 * puntos que pierden esa demanda. Delta negativo = mejora.
 */
double delta_activate(const SolutionState& solution, const Instance& instance,
                      int j, int k, double rho);


/**
 * Calcula el CAMBIO de coste de DESACTIVAR el tipo `k` en el punto `j`, SIN mutar
 * el estado. Cara "simulación" de apply_deactivate. Predice z[j]: resta C_j SOLO
 * si `k` es el ÚNICO tipo activo ahora (ningún otro active[j][t]=true), pues solo
 * entonces desactivar cierra el punto. El resto: j pierde sus bins del tipo k, y
 * los huérfanos de k se reubican con find_nearest_active (o quedan descubiertos).
 */
double delta_deactivate(const SolutionState& solution, const Instance& instance,
                        int j, int k, double rho);


/**
 * Calcula el CAMBIO de coste de un SWAP POR TIPO: mover el tipo `k` de `j_out` a
 * `j_in` (desactivar (j_out,k) + activar (j_in,k)) en un solo movimiento, SIN
 * mutar el estado. Precondición: active[j_out][k]=true, active[j_in][k]=false.
 *
 * Predice el DOBLE flip z[j]: resta C_jout si k es el único tipo activo de j_out
 * (se cierra) y suma C_jin si j_in estaba cerrado (se abre). Reasigna los edificios
 * de (j_out,k) tratando j_in como activo, y contabiliza el rebalanceo de bins.
 * Igual que delta_swap (entero) evita el doble conteo (Bug 1): los edificios ya en
 * j_out se cuentan una sola vez. Delta negativo = mejora.
 */
double delta_swap_type(const SolutionState& solution, const Instance& instance,
                       int j_out, int j_in, int k, double rho);