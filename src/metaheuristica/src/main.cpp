#include <iostream>
#include "tabu/instancia.hpp"

int main(int argc, char** argv) {
    // Ruta a la instancia de 500m (se puede sobrescribir por argumento).
    // La ruta por defecto es relativa a la raíz del proyecto.
    const std::string path = (argc > 1)
        ? argv[1]
        : "data/processed/instancia_laguna_500m.json";

    Instance inst = load_instance(path);
    preprocess(inst);

    // Verificación: para el edificio 0, tipo 0, cuántos candidatos válidos y el más cercano
    int i = 0, k = 0, k2 = 3;
    const auto& list = inst.valid_candidates[i][k];
    std::cout << "Edificio " << i << ", tipo " << k << ": "
            << list.size() << " candidatos validos\n";
    if (!list.empty()) {
        std::cout << "  Mas cercano: candidato " << list[0].j
                << " a " << list[0].distance << " m\n";
        std::cout << "  Mas lejano:  candidato " << list.back().j
                << " a " << list.back().distance << " m\n";
    }
    std::cout << "Edificio " << i << ", tipo " << k2 << ": "
            << inst.valid_candidates[i][k2].size() << " candidatos validos\n";
    if (!inst.valid_candidates[i][k2].empty()) {
        std::cout << "  Mas cercano: candidato " << inst.valid_candidates[i][k2][0].j
                << " a " << inst.valid_candidates[i][k2][0].distance << " m\n";
        std::cout << "  Mas lejano:  candidato " << inst.valid_candidates[i][k2].back().j
                << " a " << inst.valid_candidates[i][k2].back().distance << " m\n";
    }

    // Contar entradas totales en ambas estructuras: deben coincidir.
    long total_direct = 0;
    for (int i = 0; i < inst.n_buildings; ++i)
        for (int k = 0; k < inst.n_waste_types; ++k)
            total_direct += inst.valid_candidates[i][k].size();

    long total_inverse = 0;
    for (int j = 0; j < inst.n_candidates; ++j)
        total_inverse += inst.buildings_of[j].size();

    std::cout << "Total entradas valid_candidates: " << total_direct << "\n";
    std::cout << "Total entradas edificios_de:     " << total_inverse << "\n";
    std::cout << "¿Coinciden? " << (total_direct == total_inverse ? "SI" : "NO") << "\n";

    std::cout << "Candidato 42 sirve a " << inst.buildings_of[42].size() << " pares (i,k)\n";

    std::cout << "Demanda edificio 0, tipo 0: " << inst.demand[0][0] << " L/dia\n";
    std::cout << "Demanda edificio 0, tipo 3: " << inst.demand[0][3] << " L/dia\n";

    return 0;
}
