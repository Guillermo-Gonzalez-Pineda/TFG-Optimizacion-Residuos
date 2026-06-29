// Carga de instancias del problema de localización de puntos de recogida.
//
// Réplica fiel de load_instance (src/python/instancia.py): lee EXACTAMENTE los
// mismos campos del JSON, en la misma estructura, hacia el struct Instance.
// Solo se carga la instancia CRUDA — sin estructuras derivadas.

#include "tabu/instancia.hpp"

#include "nlohmann/json.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Helper: convierte un objeto JSON indexado por string {"0": v0, "1": v1, ...}
// en un std::vector<double> indexado por entero, donde vec[k] = vk.
// El vector se redimensiona primero a `n` (n_waste_types) y se rellena por
// clave, igual que el Python hace {int(k): v for k, v in obj.items()}.
// ---------------------------------------------------------------------------
static std::vector<double> object_to_vector(const json& obj, int n) {
    std::vector<double> v(n, 0.0);
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        const int k = std::stoi(it.key());   // la clave es un string ("0", "1", ...)
        v[k] = it.value().get<double>();
    }
    return v;
}

// ---------------------------------------------------------------------------
Instance load_instance(const std::string& path) {
    // --- Abrir el fichero ---
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error(
            "load_instance: no se pudo abrir el fichero de instancia: " + path);
    }

    // --- Parsear el JSON (lanza nlohmann::json::parse_error si está corrupto) ---
    const json data = json::parse(file);

    Instance inst;

    // --- Metadatos: data["meta"] ---
    const json& meta = data.at("meta");
    inst.n_buildings   = meta.at("n_buildings").get<int>();
    inst.n_candidates  = meta.at("n_candidates").get<int>();
    inst.n_waste_types = meta.at("n_waste_types").get<int>();

    // --- Parámetros del modelo: data["parameters"] ---
    // El struct ModelParameters solo contiene un subconjunto de los parámetros
    // del JSON; cargamos únicamente esos campos (el resto se ignora a propósito).
    const json& p = data.at("parameters");
    inst.params.max_bins         = p.at("max_bins").get<int>();
    inst.params.nimby_distance   = p.at("nimby_distance").get<double>();
    inst.params.waste_per_capita = p.at("waste_per_capita").get<double>();

    // Campos indexados por tipo de residuo k: objeto {"0":v,...} -> vector[k]=v.
    inst.params.bin_cost         = object_to_vector(p.at("bin_cost"),         inst.n_waste_types);
    inst.params.bin_capacity     = object_to_vector(p.at("bin_capacity"),     inst.n_waste_types);
    inst.params.coverage_radius  = object_to_vector(p.at("coverage_radius"),  inst.n_waste_types);
    inst.params.waste_proportion = object_to_vector(p.at("waste_proportion"), inst.n_waste_types);
    inst.params.waste_density    = object_to_vector(p.at("waste_density"),    inst.n_waste_types);

    // --- Edificios (conjunto I): data["sets"]["I"] ---
    // Cada entrada es {"idx": {osm_id, latitude, longitude, h_i}}.
    // El struct Building no guarda osm_id; lo omitimos.
    const json& setI = data.at("sets").at("I");
    inst.buildings.assign(inst.n_buildings, Building{});
    for (auto it = setI.begin(); it != setI.end(); ++it) {
        const int idx = std::stoi(it.key());   // índice interno del edificio
        const json& b = it.value();
        Building building;
        building.latitude  = b.at("latitude").get<double>();
        building.longitude = b.at("longitude").get<double>();
        building.h_i = b.at("h_i").get<double>();

        inst.buildings[idx] = building;
    }

    // --- Candidatos (conjunto J): data["sets"]["J"] ---
    // Cada entrada es {"idx": {osm_id, latitude, longitude, context, opening_cost}}.
    // El struct Candidate no guarda osm_id ni context; los omitimos.
    const json& setJ = data.at("sets").at("J");
    inst.candidates.assign(inst.n_candidates, Candidate{});
    for (auto it = setJ.begin(); it != setJ.end(); ++it) {
        const int idx = std::stoi(it.key());   // índice interno del candidato
        const json& c = it.value();
        Candidate cand;
        cand.latitude  = c.at("latitude").get<double>();
        cand.longitude = c.at("longitude").get<double>();
        cand.opening_cost = c.value("opening_cost", 4000.0);

        inst.candidates[idx] = cand;
    }

    // --- Distancias: data["distances"] ---
    // Estructura del JSON: { "j": { "i": distancia } }.
    // La clave EXTERNA es el candidato j y la INTERNA el edificio i, exactamente
    // como en el Python (dij[clave_externa][clave_interna], sin transponer) y
    // como espera el struct: dist[j] = mapa {edificio_i -> distancia} en metros.
    const json& dists = data.at("distances");
    inst.dist.assign(inst.n_candidates, std::unordered_map<int, double>{});
    for (auto itj = dists.begin(); itj != dists.end(); ++itj) {
        const int j = std::stoi(itj.key());           // candidato (índice del vector)
        const json& row = itj.value();
        std::unordered_map<int, double>& map_ = inst.dist[j];
        for (auto iti = row.begin(); iti != row.end(); ++iti) {
            const int i = std::stoi(iti.key());        // edificio (clave del mapa)
            map_[i] = iti.value().get<double>();
        }
    }

    return inst;
}


// ---------------------------------------------------------------------------
// Rellena las estructuras derivadas a partir de la instancia cruda.
// Por ahora, solo la Derivada 1: valid_candidates[i][k], ordenada por cercanía.
// ---------------------------------------------------------------------------
void preprocess(Instance& inst) {
    const int I  = inst.n_buildings;
    const int K  = inst.n_waste_types;
    const double r0 = inst.params.nimby_distance;

    // Dimensionar valid_candidates a [I][K][lista vacía].
    inst.valid_candidates.assign(I, std::vector<std::vector<ValidCandidate>>(K));

    // Recorrer por candidato j (como están las distancias), escribir por edificio i.
    for (int j = 0; j < inst.n_candidates; ++j) {
        for (const auto& [i, d] : inst.dist[j]) {
            if (d < r0) continue;                       // NIMBY: común a todos los tipos
            for (int k = 0; k < K; ++k) {
                if (d <= inst.params.coverage_radius[k]) {
                    inst.valid_candidates[i][k].push_back( ValidCandidate{ d, j } );
                }
            }
        }
    }

    // Ordenar cada lista de candidatos válidos por distancia ascendente.
    for (int i = 0; i < I; ++i) {
        for (int k = 0; k < K; ++k) {
            std::sort(inst.valid_candidates[i][k].begin(), inst.valid_candidates[i][k].end());
        }
    }


    // -----------------------------------------------------------------------
    // Derivada 2: la inversa. Recorremos valid_candidates (ya completa) y
    // "le damos la vuelta": cada vez que j aparece como válido para (i,k),
    // registramos (i,k) en buildings_of[j].
    // -----------------------------------------------------------------------
    inst.buildings_of.assign(inst.n_candidates, std::vector<BuildingType>{});

    for (int i = 0; i < I; ++i) {
        for (int k = 0; k < K; ++k) {
            for (const ValidCandidate& vc : inst.valid_candidates[i][k]) {
                inst.buildings_of[vc.j].push_back( BuildingType{ i, k, vc.distance } );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Derivada 3: demanda precalculada. Fórmula lineal en h_i, constante.
    //   demand[i][k] = h_i * waste_per_capita * waste_proportion[k] / waste_density[k]
    // Es la misma que compute_demand del Python, calculada de una vez.
    // -----------------------------------------------------------------------
    inst.demand.assign(I, std::vector<double>(K, 0.0));

    for (int i = 0; i < I; ++i) {
        const double h = inst.buildings[i].h_i;
        for (int k = 0; k < K; ++k) {
            inst.demand[i][k] =
                h * inst.params.waste_per_capita
                  * inst.params.waste_proportion[k]
                  / inst.params.waste_density[k];
        }
    }
}
