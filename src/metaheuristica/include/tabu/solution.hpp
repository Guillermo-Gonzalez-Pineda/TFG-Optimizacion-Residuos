#pragma once

#include <vector>
#include "tabu/instancia.hpp"   // necesita conocer la Instance

// Una solución concreta: qué puntos están abiertos y todo lo que se deriva de ello.
// Es lo que la búsqueda tabú modifica y evalúa.
struct SolutionState {
    std::vector<bool> open;                 // open[j] = ¿punto j abierto?

    // assignment[i][k] = punto asignado al edificio i para el tipo k (-1 = ninguno)
    std::vector<std::vector<int>> assignment;   

    // demand_at[j][k] = demanda tipo k que llega al punto j (litros/día)
    std::vector<std::vector<double>> demand_at; 

     // bins[j][k] = contenedores tipo k en j = techo(demand_at[j][k] / capacity[k])
    std::vector<std::vector<int>> bins;        

    // buildings_at[j] = lista de (i,k) asignados ahora a j. Inverso de assignment;
    std::vector<std::vector<std::pair<int,int>>> buildings_at;

    // --- Agregados (se actualizan en cada movimiento, no se recalculan) ---
    double total_cost;      // apertura + contenedores + penalización
    int    n_violations;    // nº de puntos con Σ_k bins[j][k] > max_bins
};