#pragma once

#include <string>
#include <vector>
#include <unordered_map>

struct ModelParameters {
    int    max_bins;          // N_j: contenedores máximo por punto (8)
    double nimby_distance;    // r_0: distancia mínima NIMBY
    double waste_per_capita;  // generación de residuo por persona

    // Indexados por tipo de residuo k = {0,1,2,3}
    std::vector<double> bin_cost;          // c_k: coste de un contenedor tipo k
    std::vector<double> bin_capacity;      // Q_k: capacidad de un contenedor tipo k
    std::vector<double> coverage_radius;   // r_k: radio de cobertura del tipo k
    std::vector<double> waste_proportion;  // proporción de residuo tipo k
    std::vector<double> waste_density;     // densidad (kg/L) del tipo k
};

struct Building {
    double latitude;
    double longitude;
    double h_i;             // nº de habitantes del edificio
};

struct Candidate {
    double latitude;
    double longitude;
    double opening_cost;  // C_j: coste de abrir este punto
};

struct ValidCandidate {
    double distance;    // distancia de j al edificio i (metros)
    int    j;           // índice del candidato
    bool operator<(const ValidCandidate& other) const {
        return distance < other.distance;
    }
};


struct BuildingType {
    int    i;           // edificio
    int    k;           // tipo de residuo
    double distance;    // dist de j a i (metros)
};

/**
 * Representa una instancia del problema de optimización de residuos.
 * Contiene todos los datos necesarios para evaluar soluciones.
 */
struct Instance {
    int n_buildings;   // nº de edificios (|I|)
    int n_candidates;  // nº de candidatos (|J|)
    int n_waste_types;  // nº de tipos de residuo (|K| = 4)

    std::vector<Building>  buildings;   // I: indexado 0..n_buildings-1
    std::vector<Candidate> candidates;  // J: indexado 0..n_candidates-1

    ModelParameters params;

    // Distancias dispersas: dist[j] = mapa de {edificio_i -> distancia}
    std::vector<std::unordered_map<int, double>> dist;

    std::vector<std::vector<std::vector<ValidCandidate>>> valid_candidates;
    //     [i]         [k]         [lista ordenada]
    std::vector<std::vector<BuildingType>> buildings_of;
    //     [j]         [lista de (i,k,dist)]
    std::vector<std::vector<double>> demand;
    //     [i]         [k]
};

/**
 * Carga una instancia CRUDA desde un fichero JSON.
 * @param path Ruta al fichero JSON.
 * @return Instancia cargada.
 */
Instance load_instance(const std::string& path);

/**
 * Preprocesa la instancia para calcular distancias, candidatos válidos y demanda.
 * @param inst Instancia a preprocesar.
 */
void preprocess(Instance& inst);
