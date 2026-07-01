// Guardado de soluciones a JSON.
//
// Escribe una solución en disco con dos niveles: un RESUMEN compatible con el
// formato del exacto (solucion_exacta_resumen.json) y un DETALLE completo con
// la asignación de edificios y los contenedores/demanda de cada punto abierto.

#include "tabu/io.hpp"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

void save_solution(const SolutionState& solution, const Instance& instance,
                   const std::string& output_path, double runtime) {
  const int n_buildings   = instance.n_buildings;
  const int n_candidates  = instance.n_candidates;
  const int n_waste_types = instance.n_waste_types;

  json out;

  // -------------------------------------------------------------------------
  // 1) RESUMEN
  // -------------------------------------------------------------------------
  out["cost"]         = solution.total_cost;
  out["runtime"]      = runtime;
  out["radius"]       = instance.osm_radius;   // trazabilidad: sobre qué instancia se computó
  out["n_violations_capacity"] = solution.n_violations_capacity;
  out["n_violations_coverage"] = solution.n_violations_coverage;

  // Puntos abiertos (en orden ascendente) y conteo.
  json open_points = json::array();
  int  n_points_open = 0;
  for (int j = 0; j < n_candidates; ++j) {
    if (solution.is_open(j)) {
      open_points.push_back(j);
      ++n_points_open;
    }
  }
  out["n_points_open"] = n_points_open;

  // Total de contenedores y desglose por tipo.
  long long total_bins = 0;
  std::vector<long long> bins_by_type(n_waste_types, 0);
  for (int j = 0; j < n_candidates; ++j) {
    for (int k = 0; k < n_waste_types; ++k) {
      const int b = solution.bins[j][k];
      total_bins        += b;
      bins_by_type[k]   += b;
    }
  }
  out["total_bins"] = total_bins;

  // Claves string por tipo ("0", "1", ...) igual que en el resumen del exacto.
  json bins_per_type = json::object();
  for (int k = 0; k < n_waste_types; ++k) {
    bins_per_type[std::to_string(k)] = bins_by_type[k];
  }
  out["bins_per_type"] = bins_per_type;

  out["open_points"] = open_points;

  // -------------------------------------------------------------------------
  // 2) DETALLE COMPLETO
  // -------------------------------------------------------------------------

  // Asignación: solo pares (i, k) con asignación != -1.
  json assignment = json::object();
  for (int i = 0; i < n_buildings; ++i) {
    json per_building = json::object();
    for (int k = 0; k < n_waste_types; ++k) {
      const int point = solution.assignment[i][k];
      if (point != -1) {
        per_building[std::to_string(k)] = point;
      }
    }
    // Solo añadimos el edificio si tiene alguna asignación.
    if (!per_building.empty()) {
      assignment[std::to_string(i)] = per_building;
    }
  }
  out["assignment"] = assignment;

  // Contenedores y demanda por cada punto abierto, desglosados por tipo.
  json bins_detail   = json::object();
  json demand_detail = json::object();
  for (int j = 0; j < n_candidates; ++j) {
    if (!solution.is_open(j)) {
      continue;
    }
    const std::string key = std::to_string(j);
    json per_point_bins   = json::object();
    json per_point_demand = json::object();
    for (int k = 0; k < n_waste_types; ++k) {
      per_point_bins[std::to_string(k)]   = solution.bins[j][k];
      per_point_demand[std::to_string(k)] = solution.demand_at[j][k];
    }
    bins_detail[key]   = per_point_bins;
    demand_detail[key] = per_point_demand;
  }
  out["bins_detail"]   = bins_detail;
  out["demand_detail"] = demand_detail;

  // -------------------------------------------------------------------------
  // Escritura a disco (creando el directorio de salida si hace falta).
  // -------------------------------------------------------------------------
  const std::filesystem::path path(output_path);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }

  std::ofstream file(output_path);
  if (!file) {
    throw std::runtime_error("No se pudo abrir el fichero de salida: " + output_path);
  }
  file << out.dump(2) << '\n';
}
