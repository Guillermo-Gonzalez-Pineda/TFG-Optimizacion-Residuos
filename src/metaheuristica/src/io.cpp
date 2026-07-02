// Guardado de soluciones a JSON, en el MISMO formato que consume el pipeline de
// análisis en Python (analisis.carga.cargar_solucion), espejo del exacto y la
// lagrangiana: z/x/w/y_assign para TODOS los índices, anidado {"j":{"k":v}} con
// claves string y valores enteros. Así las comparativas entre métodos son
// uniformes (el consumidor manda: mismo esquema de campos que el exacto).

#include "tabu/io.hpp"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

void save_solution(const SolutionState& solution, const Instance& instance,
                   const std::string& output_path, double runtime,
                   const std::string& mode) {
  const int n_buildings   = instance.n_buildings;
  const int n_candidates  = instance.n_candidates;
  const int n_waste_types = instance.n_waste_types;

  json out;
  out["metodo"]  = "tabu";
  out["cost"]    = solution.total_cost;
  out["runtime"] = runtime;
  out["modo"]    = mode;

  // z: TODOS los candidatos, plano {"j": 0|1} (= is_open(j)).
  json z = json::object();
  for (int j = 0; j < n_candidates; ++j) {
    z[std::to_string(j)] = solution.is_open(j) ? 1 : 0;
  }
  out["z"] = z;

  // x (bins) y w (active): TODOS los candidatos y tipos, anidado {"j":{"k":v}}.
  // Los puntos cerrados van con bins=0 y w=0 (misma cobertura que el exacto).
  json x = json::object();
  json w = json::object();
  for (int j = 0; j < n_candidates; ++j) {
    const std::string jk = std::to_string(j);
    json xj = json::object();
    json wj = json::object();
    for (int k = 0; k < n_waste_types; ++k) {
      const std::string kk = std::to_string(k);
      xj[kk] = solution.bins[j][k];
      wj[kk] = solution.active[j][k] ? 1 : 0;
    }
    x[jk] = xj;
    w[jk] = wj;
  }
  out["x"] = x;
  out["w"] = w;

  // y_assign: TODOS los edificios y tipos, anidado {"i":{"k":punto}}.
  // assignment[i][k] ya vale -1 cuando el par (i,k) quedó sin asignar.
  json y_assign = json::object();
  for (int i = 0; i < n_buildings; ++i) {
    const std::string ik = std::to_string(i);
    json yi = json::object();
    for (int k = 0; k < n_waste_types; ++k) {
      yi[std::to_string(k)] = solution.assignment[i][k];
    }
    y_assign[ik] = yi;
  }
  out["y_assign"] = y_assign;

  // Escritura a disco (creando el directorio de salida si hace falta).
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
