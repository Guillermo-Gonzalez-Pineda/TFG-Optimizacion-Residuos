#pragma once

#include "tabu/solution.hpp"
#include "tabu/instancia.hpp"

/**
 * Construye una solución inicial mediante un greedy de cobertura con control
 * de capacidad. 
 * Fase 1: abre puntos hasta cubrir todos los pares (edificio, tipo).
 * Fase 2: abre puntos adicionales para aliviar la saturación.
 *
 * La solución resultante queda en `solution`, con sus agregados calculados.
 *
 * @param solution Solución a construir (se sobrescribe desde cero).
 * @param instance Instancia del problema.
 * @param rho Peso de la penalización por violación (para el coste final).
 */
void construct_initial(SolutionState& solution, const Instance& instance, double rho);