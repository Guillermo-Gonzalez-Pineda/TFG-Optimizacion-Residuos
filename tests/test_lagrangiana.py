"""
Tests for lagrangiana.precompute_valid_candidates.

Structure:
  1. TestShape          — dimensiones de la salida
  2. TestInvariants     — propiedades matemáticas del modelo HDM
  3. TestEdgeCases      — instancias sintéticas para casos límite
  4. TestRegression     — valores fijados sobre instancia_laguna.json
  5. TestDataclasses    — construcción de LRSolution / Multipliers / etc.

Instancia de referencia: data/processed/instancia_laguna.json
  · 740 edificios, 133 candidatos, 4 tipos de residuo
  · dij[j_idx][i_idx] = distancia candidato j → edificio i
  · nimby_distance = 10 m
  · coverage_radius = {0: 150 m, 1: 150 m, 2: 250 m, 3: 350 m}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from instancia import (
    BuildingData,
    CandidateContext,
    CandidateData,
    Instance,
    ModelParameters,
    load_instance,
)
from instancia import compute_demand
from lagrangiana import (
    FeasibleSolution,
    LagrangianResult,
    LRSolution,
    Multipliers,
    precompute_valid_candidates,
    repair_solution,
    solve_plr_x,
    solve_plr_yw,
    solve_plr_z,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parents[1]
INSTANCE_PATH = ROOT / "data" / "processed" / "instancia_laguna.json"


# ---------------------------------------------------------------------------
# Helpers for synthetic instances
# ---------------------------------------------------------------------------

def _make_params(
    nimby: float = 10.0,
    cov: tuple[float, ...] = (150.0, 150.0, 250.0, 350.0),
) -> ModelParameters:
    n_k = len(cov)
    wp = {0: 0.5012, 1: 0.0791, 2: 0.3885, 3: 0.0312}
    return ModelParameters(
        opening_cost=4000.0,
        max_bins=8,
        nimby_distance=nimby,
        waste_per_capita=1.32,
        overflow_penalty=500.0,
        bin_cost={k: 300.0 for k in range(n_k)},
        bin_capacity={k: 120.0 for k in range(n_k)},
        coverage_radius={k: cov[k] for k in range(n_k)},
        waste_proportion={k: wp.get(k, 1.0 / n_k) for k in range(n_k)},
        collection_frequency={k: 1.0 for k in range(n_k)},
        lognormal_mu=0.0,
        lognormal_sigma=0.25,
        overflow_threshold=0.05,
    )


def _make_instance(
    I: dict,
    J: dict,
    dij: dict,
    K: list[int] | None = None,
    params: ModelParameters | None = None,
) -> Instance:
    """Build a minimal Instance suitable for unit testing."""
    if K is None:
        K = [0, 1, 2, 3]
    if params is None:
        params = _make_params()
    return Instance(
        study_case="test",
        osm_radius_m=500,
        dijkstra_radius_m=350,
        generated_at="2026-01-01T00:00:00",
        references=[],
        n_buildings=len(I),
        n_candidates=len(J),
        n_waste_types=len(K),
        total_population=100,
        n_dijkstra_connections=sum(len(v) for v in dij.values()),
        i_to_idx={},
        idx_to_i={},
        j_to_idx={},
        idx_to_j={},
        K=K,
        I=I,
        J=J,
        dij=dij,
        params=params,
    )


def _bld(i: int, h_i: float = 100.0) -> BuildingData:
    return BuildingData(osm_id=str(i), latitude=0.0, longitude=0.0, h_i=h_i)


def _cand(j: int) -> CandidateData:
    return CandidateData(osm_id=str(j), latitude=0.0, longitude=0.0)


# ---------------------------------------------------------------------------
# Fixtures  (scope=module → la instancia real se carga una sola vez)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def laguna() -> Instance:
    return load_instance(str(INSTANCE_PATH))


@pytest.fixture(scope="module")
def vc(laguna: Instance):
    return precompute_valid_candidates(laguna)


# ===========================================================================
# 1. Tests de dimensiones
# ===========================================================================

class TestShape:
    def test_outer_length_equals_n_buildings(self, laguna, vc):
        assert len(vc) == laguna.n_buildings

    def test_inner_length_equals_n_waste_types(self, laguna, vc):
        n_k = laguna.n_waste_types
        for i in range(laguna.n_buildings):
            assert len(vc[i]) == n_k, (
                f"building {i}: esperado {n_k} listas, obtenido {len(vc[i])}"
            )

    def test_every_element_is_list(self, laguna, vc):
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                assert isinstance(vc[i][k], list), (
                    f"vc[{i}][{k}] es {type(vc[i][k]).__name__}, se esperaba list"
                )

    def test_candidate_indices_within_valid_range(self, laguna, vc):
        n_j = laguna.n_candidates
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                for j in vc[i][k]:
                    assert 0 <= j < n_j, (
                        f"Índice de candidato fuera de rango: j={j} en vc[{i}][{k}]"
                    )


# ===========================================================================
# 2. Tests de invariantes matemáticos
# ===========================================================================

class TestInvariants:
    def test_nimby_constraint_respected(self, laguna, vc):
        """d_ij >= r_0 (NIMBY) para todo (i, k, j) en vc."""
        nimby = laguna.params.nimby_distance
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                for j in vc[i][k]:
                    d = laguna.dij[j][i]
                    assert d >= nimby, (
                        f"NIMBY violado: i={i} k={k} j={j} d={d:.3f} < r_0={nimby}"
                    )

    def test_coverage_radius_respected(self, laguna, vc):
        """d_ij <= r_k para todo (i, k, j) en vc."""
        cov = laguna.params.coverage_radius
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                r_k = cov[k]
                for j in vc[i][k]:
                    d = laguna.dij[j][i]
                    assert d <= r_k, (
                        f"Cobertura violada: i={i} k={k} j={j} d={d:.3f} > r_k={r_k}"
                    )

    def test_sorted_ascending_by_distance(self, laguna, vc):
        """La lista de candidatos de cada (i, k) está ordenada por distancia creciente."""
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                lst = vc[i][k]
                if len(lst) < 2:
                    continue
                distances = [laguna.dij[j][i] for j in lst]
                for a, b in zip(distances, distances[1:]):
                    assert a <= b, (
                        f"No ordenado: i={i} k={k} distancias={[round(x, 2) for x in distances[:6]]}"
                    )

    def test_equal_coverage_radius_yields_identical_lists(self, laguna, vc):
        """k=0 y k=1 comparten r_k=150 m → sus listas de candidatos deben ser idénticas."""
        cov = laguna.params.coverage_radius
        if cov[0] != cov[1]:
            pytest.skip("k=0 y k=1 no tienen el mismo radio en esta instancia")
        for i in range(laguna.n_buildings):
            assert vc[i][0] == vc[i][1], (
                f"i={i}: listas k=0 y k=1 difieren pese a tener el mismo radio"
            )

    def test_wider_radius_more_or_equal_candidates(self, laguna, vc):
        """k=3 (r=350 m) debe tener al menos tantos candidatos como k=0 (r=150 m)."""
        for i in range(laguna.n_buildings):
            assert len(vc[i][0]) <= len(vc[i][3]), (
                f"i={i}: k=0 tiene más candidatos que k=3 "
                f"({len(vc[i][0])} > {len(vc[i][3])})"
            )

    def test_candidates_k0_subset_of_k3(self, laguna, vc):
        """Todo candidato válido para k=0 (r=150) es válido para k=3 (r=350)."""
        for i in range(laguna.n_buildings):
            set_k0 = set(vc[i][0])
            set_k3 = set(vc[i][3])
            diff = set_k0 - set_k3
            assert not diff, (
                f"i={i}: candidatos de k=0 no contenidos en k=3: {diff}"
            )

    def test_candidates_k0_subset_of_k2(self, laguna, vc):
        """Todo candidato válido para k=0 (r=150) es válido para k=2 (r=250)."""
        for i in range(laguna.n_buildings):
            set_k0 = set(vc[i][0])
            set_k2 = set(vc[i][2])
            diff = set_k0 - set_k2
            assert not diff, (
                f"i={i}: candidatos de k=0 no contenidos en k=2: {diff}"
            )

    def test_candidates_k2_subset_of_k3(self, laguna, vc):
        """Todo candidato válido para k=2 (r=250) es válido para k=3 (r=350)."""
        for i in range(laguna.n_buildings):
            set_k2 = set(vc[i][2])
            set_k3 = set(vc[i][3])
            diff = set_k2 - set_k3
            assert not diff, (
                f"i={i}: candidatos de k=2 no contenidos en k=3: {diff}"
            )

    def test_completeness_no_valid_pair_omitted(self, laguna, vc):
        """Para todo (j, i) en dij que satisface las restricciones, j ∈ vc[i][k]."""
        nimby = laguna.params.nimby_distance
        cov = laguna.params.coverage_radius
        # Preconstruir sets para O(1) lookup
        vc_sets = [
            [set(vc[i][k]) for k in laguna.K]
            for i in range(laguna.n_buildings)
        ]
        for j_idx, i_dict in laguna.dij.items():
            for i_idx, d in i_dict.items():
                for k in laguna.K:
                    should = nimby <= d <= cov[k]
                    actually = j_idx in vc_sets[i_idx][k]
                    assert should == actually, (
                        f"Completitud violada: j={j_idx} i={i_idx} k={k} "
                        f"d={d:.2f} debería={should} incluido={actually}"
                    )

    def test_no_duplicate_candidates_per_building_type(self, laguna, vc):
        """Ningún candidato debe aparecer dos veces en vc[i][k]."""
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                lst = vc[i][k]
                assert len(lst) == len(set(lst)), (
                    f"Duplicados en vc[{i}][{k}]: {lst}"
                )


# ===========================================================================
# 3. Tests de casos límite (instancias sintéticas)
# ===========================================================================

class TestEdgeCases:
    def test_all_candidates_too_close(self):
        """Todos los candidatos están dentro del radio NIMBY → listas vacías."""
        I = {0: _bld(0)}
        J = {0: _cand(0), 1: _cand(1)}
        dij = {0: {0: 5.0}, 1: {0: 8.0}}   # ambos < nimby=10
        inst = _make_instance(I, J, dij)
        result = precompute_valid_candidates(inst)
        for k in range(4):
            assert result[0][k] == [], f"k={k}: esperado [], obtenido {result[0][k]}"

    def test_all_candidates_too_far(self):
        """Todos los candidatos exceden el radio máximo de cobertura → listas vacías."""
        I = {0: _bld(0)}
        J = {0: _cand(0), 1: _cand(1)}
        dij = {0: {0: 400.0}, 1: {0: 500.0}}  # ambos > r_3=350
        inst = _make_instance(I, J, dij)
        result = precompute_valid_candidates(inst)
        for k in range(4):
            assert result[0][k] == [], f"k={k}: esperado [], obtenido {result[0][k]}"

    def test_boundary_exactly_at_nimby_is_included(self):
        """Candidato a d = r_0 (frontera NIMBY) debe incluirse (≥ es inclusivo)."""
        I = {0: _bld(0)}
        J = {0: _cand(0)}
        nimby = 10.0
        dij = {0: {0: nimby}}
        inst = _make_instance(I, J, dij, params=_make_params(nimby=nimby))
        result = precompute_valid_candidates(inst)
        for k in range(4):
            assert 0 in result[0][k], (
                f"k={k}: candidato en frontera NIMBY ausente"
            )

    def test_boundary_just_inside_nimby_is_excluded(self):
        """Candidato a d = r_0 - ε debe excluirse (viola NIMBY)."""
        I = {0: _bld(0)}
        J = {0: _cand(0)}
        nimby = 10.0
        dij = {0: {0: nimby - 1e-6}}
        inst = _make_instance(I, J, dij, params=_make_params(nimby=nimby))
        result = precompute_valid_candidates(inst)
        for k in range(4):
            assert result[0][k] == [], f"k={k}: candidato viola NIMBY y no debería aparecer"

    def test_boundary_exactly_at_coverage_is_included(self):
        """Candidato a d = r_k (frontera cobertura) debe incluirse (≤ es inclusivo)."""
        I = {0: _bld(0)}
        J = {0: _cand(0)}
        dij = {0: {0: 150.0}}   # d == r_0 == r_1
        inst = _make_instance(I, J, dij)
        result = precompute_valid_candidates(inst)
        assert 0 in result[0][0], "d=r_0=150 debe incluirse en k=0"
        assert 0 in result[0][1], "d=r_1=150 debe incluirse en k=1"
        assert 0 in result[0][2], "d=150 ≤ r_2=250 debe incluirse en k=2"
        assert 0 in result[0][3], "d=150 ≤ r_3=350 debe incluirse en k=3"

    def test_boundary_just_outside_coverage_is_excluded(self):
        """Candidato a d = r_k + ε debe excluirse para ese k."""
        I = {0: _bld(0)}
        J = {0: _cand(0)}
        dij = {0: {0: 150.0 + 1e-6}}   # justo por encima de r_0 = r_1
        inst = _make_instance(I, J, dij)
        result = precompute_valid_candidates(inst)
        assert result[0][0] == [], "d > r_0=150 → excluido de k=0"
        assert result[0][1] == [], "d > r_1=150 → excluido de k=1"
        assert 0 in result[0][2], "d=150+ε ≤ r_2=250 → incluido en k=2"
        assert 0 in result[0][3], "d=150+ε ≤ r_3=350 → incluido en k=3"

    def test_empty_dij_yields_all_empty(self):
        """Sin conexiones dij, todas las listas deben ser vacías."""
        I = {0: _bld(0), 1: _bld(1)}
        J = {0: _cand(0)}
        dij = {}
        inst = _make_instance(I, J, dij)
        result = precompute_valid_candidates(inst)
        for i in range(2):
            for k in range(4):
                assert result[i][k] == [], f"vc[{i}][{k}] debe ser [] sin conexiones"

    def test_single_building_single_candidate_in_range(self):
        """1 edificio, 1 candidato en rango válido → aparece en todos los k."""
        I = {0: _bld(0)}
        J = {0: _cand(0)}
        dij = {0: {0: 100.0}}   # nimby=10 ≤ 100 ≤ r_k para todo k
        inst = _make_instance(I, J, dij)
        result = precompute_valid_candidates(inst)
        for k in range(4):
            assert result[0][k] == [0], f"k={k}: esperado [0], obtenido {result[0][k]}"

    def test_distance_only_valid_for_wider_k(self):
        """d=200 m: excluido de k=0,1 (r=150); incluido en k=2 (r=250), k=3 (r=350)."""
        I = {0: _bld(0)}
        J = {0: _cand(0)}
        dij = {0: {0: 200.0}}
        inst = _make_instance(I, J, dij)
        result = precompute_valid_candidates(inst)
        assert result[0][0] == [], "d=200 > r_0=150"
        assert result[0][1] == [], "d=200 > r_1=150"
        assert result[0][2] == [0], "d=200 ≤ r_2=250"
        assert result[0][3] == [0], "d=200 ≤ r_3=350"

    def test_multiple_candidates_sorted_ascending(self):
        """4 candidatos a distancias desordenadas → la lista de salida es ASC."""
        I = {0: _bld(0)}
        J = {j: _cand(j) for j in range(4)}
        # candidatos j=0..3 a distancias 100, 50, 130, 80 — todos en [10, 150]
        dij = {0: {0: 100.0}, 1: {0: 50.0}, 2: {0: 130.0}, 3: {0: 80.0}}
        inst = _make_instance(I, J, dij)
        result = precompute_valid_candidates(inst)
        # Orden esperado: j=1(50) < j=3(80) < j=0(100) < j=2(130)
        assert result[0][0] == [1, 3, 0, 2], (
            f"k=0: esperado [1,3,0,2], obtenido {result[0][0]}"
        )

    def test_building_with_zero_demand_still_gets_candidates(self):
        """h_i=0 no afecta a precompute_valid_candidates (no usa demanda)."""
        I = {0: _bld(0, h_i=0)}
        J = {0: _cand(0)}
        dij = {0: {0: 50.0}}   # distancia válida
        inst = _make_instance(I, J, dij)
        result = precompute_valid_candidates(inst)
        for k in range(4):
            assert 0 in result[0][k], f"h_i=0 no debería impedir candidatos (k={k})"

    def test_disconnected_building_gets_empty_lists(self):
        """Edificio sin ninguna conexión dij → listas vacías; el conectado sí tiene."""
        I = {0: _bld(0), 1: _bld(1)}
        J = {0: _cand(0)}
        dij = {0: {0: 50.0}}   # sólo edificio 0 conectado al candidato 0
        inst = _make_instance(I, J, dij)
        result = precompute_valid_candidates(inst)
        for k in range(4):
            assert 0 in result[0][k], f"edificio 0 debe tener candidato 0 en k={k}"
            assert result[1][k] == [], f"edificio 1 desconectado, k={k}"

    def test_single_waste_type(self):
        """K=[0] (un solo tipo) → listas internas de longitud 1."""
        I = {0: _bld(0)}
        J = {0: _cand(0)}
        dij = {0: {0: 50.0}}
        params = _make_params(cov=(150.0,))
        inst = _make_instance(I, J, dij, K=[0], params=params)
        result = precompute_valid_candidates(inst)
        assert len(result[0]) == 1, "Con K=[0] se espera 1 sublista por edificio"
        assert result[0][0] == [0]

    def test_no_candidates_in_dij_for_building(self):
        """Un candidato sin conexiones al edificio 1 → vc[1] vacío para todos los k."""
        I = {0: _bld(0), 1: _bld(1)}
        J = {0: _cand(0), 1: _cand(1)}
        # Candidato 0 conectado a edificio 0; candidato 1 conectado a edificio 0
        dij = {0: {0: 30.0}, 1: {0: 60.0}}
        inst = _make_instance(I, J, dij)
        result = precompute_valid_candidates(inst)
        for k in range(4):
            assert result[1][k] == [], f"edificio 1 sin conexiones, k={k}"


# ===========================================================================
# 4. Tests de regresión (instancia real)
# ===========================================================================

class TestRegression:
    """Valores fijados sobre instancia_laguna.json — fallar aquí indica regresión."""

    def test_instance_dimensions(self, laguna):
        assert laguna.n_buildings == 740
        assert laguna.n_candidates == 133
        assert laguna.n_waste_types == 4
        assert laguna.K == [0, 1, 2, 3]

    def test_total_valid_slots(self, laguna, vc):
        """Suma total de (edificio, tipo) con al menos un candidato asignado."""
        total = sum(
            len(vc[i][k])
            for i in range(laguna.n_buildings)
            for k in range(laguna.n_waste_types)
        )
        assert total == 45484

    def test_valid_counts_per_waste_type(self, laguna, vc):
        counts = {
            k: sum(len(vc[i][k]) for i in range(laguna.n_buildings))
            for k in range(laguna.n_waste_types)
        }
        assert counts[0] == 5629,  f"k=0: {counts[0]}"
        assert counts[1] == 5629,  f"k=1: {counts[1]}"
        assert counts[2] == 12529, f"k=2: {counts[2]}"
        assert counts[3] == 21697, f"k=3: {counts[3]}"

    def test_nearest_candidates_building0_k0(self, vc):
        """Edificio 0, tipo orgánico: candidatos más cercanos."""
        assert vc[0][0] == [31, 9], f"Obtenido: {vc[0][0]}"

    def test_nearest_distance_building0_k0(self, laguna, vc):
        """Distancia al candidato más cercano del edificio 0 para k=0."""
        j0 = vc[0][0][0]
        d = laguna.dij[j0][0]
        assert abs(d - 47.241296) < 1e-4, f"Distancia inesperada: {d}"

    def test_second_candidate_building0_k0(self, laguna, vc):
        j1 = vc[0][0][1]
        d = laguna.dij[j1][0]
        assert abs(d - 81.663704) < 1e-4, f"Distancia inesperada: {d}"

    def test_no_building_isolated_from_all_types(self, laguna, vc):
        """Ningún edificio debe carecer de candidatos para todos los tipos."""
        isolated = [
            i for i in range(laguna.n_buildings)
            if all(len(vc[i][k]) == 0 for k in range(laguna.n_waste_types))
        ]
        assert isolated == [], f"Edificios sin ningún candidato: {isolated}"

    def test_k3_has_more_total_candidates_than_k0(self, laguna, vc):
        total_k0 = sum(len(vc[i][0]) for i in range(laguna.n_buildings))
        total_k3 = sum(len(vc[i][3]) for i in range(laguna.n_buildings))
        assert total_k3 > total_k0, (
            f"k=3 ({total_k3}) debería superar k=0 ({total_k0}) en total"
        )

    def test_average_candidates_k3_greater_than_k0(self, laguna, vc):
        avg_k0 = sum(len(vc[i][0]) for i in range(laguna.n_buildings)) / laguna.n_buildings
        avg_k3 = sum(len(vc[i][3]) for i in range(laguna.n_buildings)) / laguna.n_buildings
        assert avg_k3 > avg_k0

    def test_dijkstra_connections_count(self, laguna):
        total = sum(len(v) for v in laguna.dij.values())
        assert total == laguna.n_dijkstra_connections


# ===========================================================================
# 5. Tests de construcción de dataclasses
# ===========================================================================

class TestDataclasses:
    """Verifica que los dataclasses de lagrangiana.py se construyen con las
    shapes correctas y son mutuamente coherentes."""

    def test_multipliers_shapes(self):
        n_j, n_k = 7, 4
        m = Multipliers(
            mu=np.zeros((n_j, n_k)),
            lbd=np.zeros(n_j),
            nu=np.zeros((n_j, n_k)),
        )
        assert m.mu.shape == (n_j, n_k)
        assert m.lbd.shape == (n_j,)
        assert m.nu.shape == (n_j, n_k)

    def test_multipliers_all_zero_initial(self):
        m = Multipliers(
            mu=np.zeros((3, 4)), lbd=np.zeros(3), nu=np.zeros((3, 4))
        )
        assert np.all(m.mu == 0.0)
        assert np.all(m.lbd == 0.0)
        assert np.all(m.nu == 0.0)

    def test_multipliers_are_mutable(self):
        """frozen=False → se puede modificar in-place (necesario en el subgradiente)."""
        m = Multipliers(
            mu=np.zeros((3, 4)), lbd=np.zeros(3), nu=np.zeros((3, 4))
        )
        m.mu[0, 0] = 99.0
        assert m.mu[0, 0] == 99.0

    def test_lr_solution_shapes(self):
        n_j, n_i, n_k = 10, 20, 4
        sol = LRSolution(
            z=np.zeros(n_j, dtype=bool),
            x=np.zeros((n_j, n_k), dtype=int),
            y_assign=np.zeros((n_i, n_k), dtype=int),
            w=np.zeros((n_j, n_k), dtype=bool),
            obj_plrz=0.0,
            obj_plrx=0.0,
            obj_plryw=0.0,
        )
        assert sol.z.shape == (n_j,)
        assert sol.x.shape == (n_j, n_k)
        assert sol.y_assign.shape == (n_i, n_k)
        assert sol.w.shape == (n_j, n_k)

    def test_feasible_solution_shapes(self):
        n_j, n_i, n_k = 10, 20, 4
        fs = FeasibleSolution(
            z=np.zeros(n_j, dtype=bool),
            x=np.zeros((n_j, n_k), dtype=int),
            y_assign=np.zeros((n_i, n_k), dtype=int),
            w=np.zeros((n_j, n_k), dtype=bool),
            cost=9999.0,
        )
        assert fs.z.shape == (n_j,)
        assert fs.x.shape == (n_j, n_k)
        assert fs.y_assign.shape == (n_i, n_k)
        assert fs.cost == 9999.0

    def test_lagrangian_result_gap_formula(self):
        """gap = (UB - LB) / UB, el resultado debe registrar el valor pasado."""
        lb, ub = 1000.0, 1500.0
        fs = FeasibleSolution(
            z=np.array([True]),
            x=np.ones((1, 4), dtype=int),
            y_assign=np.zeros((2, 4), dtype=int),
            w=np.ones((1, 4), dtype=bool),
            cost=ub,
        )
        result = LagrangianResult(
            best_feasible=fs,
            best_lb=lb,
            best_ub=ub,
            gap=(ub - lb) / ub,
            n_iterations=100,
            lb_history=[lb],
            ub_history=[ub],
        )
        assert result.gap == pytest.approx((ub - lb) / ub)
        assert result.best_lb < result.best_ub

    def test_lagrangian_result_histories_length(self):
        fs = FeasibleSolution(
            z=np.array([False]),
            x=np.zeros((1, 4), dtype=int),
            y_assign=np.zeros((1, 4), dtype=int),
            w=np.zeros((1, 4), dtype=bool),
            cost=0.0,
        )
        lbs = [100.0 + i for i in range(5)]
        ubs = [200.0 - i for i in range(5)]
        result = LagrangianResult(
            best_feasible=fs,
            best_lb=lbs[-1],
            best_ub=ubs[-1],
            gap=0.3,
            n_iterations=len(lbs),
            lb_history=lbs,
            ub_history=ubs,
        )
        assert len(result.lb_history) == 5
        assert len(result.ub_history) == 5
        assert result.n_iterations == 5


# ===========================================================================
# 6. Tests de solve_plr_yw
# ===========================================================================

# --- Fixtures adicionales (module-scoped para no recomputar) ---------------

@pytest.fixture(scope="module")
def mults_zero(laguna) -> Multipliers:
    n_j, n_k = laguna.n_candidates, laguna.n_waste_types
    return Multipliers(
        mu=np.zeros((n_j, n_k)),
        lbd=np.zeros(n_j),
        nu=np.zeros((n_j, n_k)),
    )


@pytest.fixture(scope="module")
def yw_zero(laguna, vc, mults_zero):
    """solve_plr_yw con todos los multiplicadores a cero."""
    return solve_plr_yw(laguna, mults_zero, vc)


@pytest.fixture(scope="module")
def yw_mu1(laguna, vc):
    """solve_plr_yw con mu=1, nu=0."""
    n_j, n_k = laguna.n_candidates, laguna.n_waste_types
    m = Multipliers(
        mu=np.ones((n_j, n_k)),
        lbd=np.zeros(n_j),
        nu=np.zeros((n_j, n_k)),
    )
    return solve_plr_yw(laguna, m, vc)


class TestSolvePlrYw:
    """
    Tests para solve_plr_yw — subproblema de asignación P_LR_yw.

    Semántica del objetivo:
        obj = positive_term - negative_term
        positive_term = Σ_{(i,k): y≠-1}  mu[y[i,k], k] · q_ik
        negative_term = max_bins · Σ_{j,k} nu[j,k]
        q_ik          = h_i · waste_per_capita · waste_proportion[k]

    Nearest-allocation: y_assign[i,k] = valid_candidates[i][k][0]  (el más cercano).
    w_jk = 1 para todo (j,k) — coeficiente siempre negativo en la relajación.
    """

    # -----------------------------------------------------------------------
    # 1. Shapes
    # -----------------------------------------------------------------------

    def test_y_assign_shape(self, laguna, yw_zero):
        ya, w, _ = yw_zero
        assert ya.shape == (laguna.n_buildings, laguna.n_waste_types), (
            f"Esperado ({laguna.n_buildings}, {laguna.n_waste_types}), obtenido {ya.shape}"
        )

    def test_w_shape(self, laguna, yw_zero):
        ya, w, _ = yw_zero
        assert w.shape == (laguna.n_candidates, laguna.n_waste_types), (
            f"Esperado ({laguna.n_candidates}, {laguna.n_waste_types}), obtenido {w.shape}"
        )

    def test_obj_is_scalar(self, yw_zero):
        _, _, obj = yw_zero
        assert np.ndim(obj) == 0, f"obj_plryw debe ser escalar, shape={np.shape(obj)}"

    # -----------------------------------------------------------------------
    # 2. w es siempre True
    # -----------------------------------------------------------------------

    def test_w_all_true_zero_multipliers(self, yw_zero):
        _, w, _ = yw_zero
        assert np.all(w), "w debe ser todo True con multiplicadores cero"

    def test_w_all_true_nonzero_multipliers(self, yw_mu1):
        _, w, _ = yw_mu1
        assert np.all(w), "w debe ser todo True con mu=1"

    def test_w_dtype_is_bool(self, yw_zero):
        _, w, _ = yw_zero
        assert w.dtype == bool, f"w debe ser dtype=bool, obtenido {w.dtype}"

    # -----------------------------------------------------------------------
    # 3. y_assign contiene solo índices válidos en [0, n_candidates) o -1
    # -----------------------------------------------------------------------

    def test_y_assign_values_valid_or_minus1(self, laguna, yw_zero):
        ya, _, _ = yw_zero
        n_j = laguna.n_candidates
        mask = ya != -1
        assert np.all(ya[mask] >= 0), "y_assign tiene valores negativos distintos de -1"
        assert np.all(ya[mask] < n_j), (
            f"y_assign tiene índices ≥ n_candidates={n_j}: max={ya[mask].max()}"
        )

    def test_y_assign_dtype_int(self, yw_zero):
        ya, _, _ = yw_zero
        assert np.issubdtype(ya.dtype, np.integer), (
            f"y_assign debe ser dtype entero, obtenido {ya.dtype}"
        )

    # -----------------------------------------------------------------------
    # 4. Nearest-allocation: j asignado es el más cercano válido
    # -----------------------------------------------------------------------

    def test_nearest_allocation_equals_vc_first(self, laguna, vc, yw_zero):
        """y_assign[i,k] debe ser vc[i][k][0] cuando hay candidatos válidos."""
        ya, _, _ = yw_zero
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                if vc[i][k]:
                    assert ya[i, k] == vc[i][k][0], (
                        f"i={i} k={k}: y_assign={ya[i,k]}, vc[i][k][0]={vc[i][k][0]}"
                    )

    def test_nearest_allocation_minimum_distance_in_dij(self, laguna, vc, yw_zero):
        """La distancia al j asignado es ≤ distancia a cualquier otro candidato válido."""
        ya, _, _ = yw_zero
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                j = ya[i, k]
                if j == -1:
                    continue
                d_assigned = laguna.dij[j][i]
                for j2 in vc[i][k]:
                    d2 = laguna.dij[j2][i]
                    assert d_assigned <= d2 + 1e-9, (
                        f"i={i} k={k}: j={j} d={d_assigned:.4f} > j2={j2} d2={d2:.4f}"
                    )

    def test_y_assign_invariant_to_multipliers(self, laguna, vc, yw_zero):
        """La asignación nearest-first no depende de los multiplicadores."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(7)
        m_rand = Multipliers(
            mu=np.random.rand(n_j, n_k),
            lbd=np.zeros(n_j),
            nu=np.random.rand(n_j, n_k),
        )
        ya_rand, _, _ = solve_plr_yw(laguna, m_rand, vc)
        ya_zero, _, _ = yw_zero
        assert np.array_equal(ya_rand, ya_zero), (
            "y_assign cambia con los multiplicadores, pero debería ser siempre nearest-first"
        )

    def test_y_minus1_iff_no_valid_candidates(self, laguna, vc, yw_zero):
        """y_assign[i,k] == -1 ↔ valid_candidates[i][k] está vacío."""
        ya, _, _ = yw_zero
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                is_minus1  = (ya[i, k] == -1)
                no_cands   = (len(vc[i][k]) == 0)
                assert is_minus1 == no_cands, (
                    f"i={i} k={k}: y=-1={is_minus1}, no_cands={no_cands} — inconsistente"
                )

    # -----------------------------------------------------------------------
    # 5. NIMBY y cobertura respetadas en todas las asignaciones
    # -----------------------------------------------------------------------

    def test_nimby_respected_in_all_assignments(self, laguna, yw_zero):
        """d_ij ≥ nimby_distance para todo (i,k) asignado."""
        ya, _, _ = yw_zero
        nimby = laguna.params.nimby_distance
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                j = ya[i, k]
                if j == -1:
                    continue
                d = laguna.dij[j][i]
                assert d >= nimby, (
                    f"NIMBY violado: i={i} k={k} j={j} d={d:.3f} < r_0={nimby}"
                )

    def test_coverage_respected_in_all_assignments(self, laguna, yw_zero):
        """d_ij ≤ coverage_radius[k] para todo (i,k) asignado."""
        ya, _, _ = yw_zero
        cov = laguna.params.coverage_radius
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                j = ya[i, k]
                if j == -1:
                    continue
                d = laguna.dij[j][i]
                assert d <= cov[k], (
                    f"Cobertura violada: i={i} k={k} j={j} d={d:.3f} > r_k={cov[k]}"
                )

    # -----------------------------------------------------------------------
    # 6. obj_plryw = 0 cuando todos los multiplicadores son cero
    # -----------------------------------------------------------------------

    def test_obj_zero_with_zero_multipliers(self, yw_zero):
        """Con mu=0 y nu=0 (primera iteración): obj = 0·q - max_bins·0 = 0."""
        _, _, obj = yw_zero
        assert obj == 0.0, f"Esperado 0.0, obtenido {obj}"

    # -----------------------------------------------------------------------
    # 7. Edificio sin candidatos válidos → y_assign = -1
    # -----------------------------------------------------------------------

    def test_no_candidates_yields_minus1(self):
        """Instancia sintética sin conexiones dij → todos y_assign = -1."""
        inst = load_instance(str(INSTANCE_PATH))
        # Construimos instancia con 2 edificios y 1 candidato pero dij vacío
        I = {0: inst.I[0], 1: inst.I[1]}
        J = {0: inst.J[0]}
        mini = _make_instance(I, J, dij={})
        vc_mini = precompute_valid_candidates(mini)
        n_j, n_k = 1, 4
        m = Multipliers(
            mu=np.ones((n_j, n_k)),
            lbd=np.zeros(n_j),
            nu=np.ones((n_j, n_k)),
        )
        ya, _, _ = solve_plr_yw(mini, m, vc_mini)
        assert np.all(ya == -1), (
            f"Esperado todo -1 sin candidatos válidos, obtenido:\n{ya}"
        )

    def test_no_candidates_obj_only_negative_term(self):
        """Sin candidatos, positive_term=0 → obj = -max_bins·nu.sum()."""
        inst = load_instance(str(INSTANCE_PATH))
        I = {0: inst.I[0]}
        J = {0: inst.J[0]}
        mini = _make_instance(I, J, dij={})
        vc_mini = precompute_valid_candidates(mini)
        nu_val = 3.0
        n_j, n_k = 1, 4
        m = Multipliers(
            mu=np.ones((n_j, n_k)),
            lbd=np.zeros(n_j),
            nu=np.full((n_j, n_k), nu_val),
        )
        _, _, obj = solve_plr_yw(mini, m, vc_mini)
        expected = -mini.params.max_bins * nu_val * n_j * n_k
        assert obj == pytest.approx(expected), (
            f"Esperado {expected}, obtenido {obj}"
        )

    def test_single_building_single_candidate_assignment(self):
        """1 edificio, 1 candidato a distancia válida: asignación y desglose exactos."""
        inst = load_instance(str(INSTANCE_PATH))
        h = 50.0
        I = {0: _bld(0, h_i=h)}
        J = {0: _cand(0)}
        dij = {0: {0: 80.0}}   # válido para todos los k
        mini = _make_instance(I, J, dij)
        vc_mini = precompute_valid_candidates(mini)
        n_j, n_k = 1, 4
        mu_val, nu_val = 2.0, 1.5
        m = Multipliers(
            mu=np.full((n_j, n_k), mu_val),
            lbd=np.zeros(n_j),
            nu=np.full((n_j, n_k), nu_val),
        )
        ya, w, obj = solve_plr_yw(mini, m, vc_mini)

        assert np.all(ya == 0), f"Esperado todo 0, obtenido {ya}"
        assert np.all(w), "w debe ser todo True"

        # Positive term: suma sobre k de mu_val * q_ik
        expected_pos = sum(
            mu_val * compute_demand(h, mini.params, k) for k in range(n_k)
        )
        expected_neg = mini.params.max_bins * nu_val * n_j * n_k
        expected_obj = expected_pos - expected_neg
        assert obj == pytest.approx(expected_obj, rel=1e-9), (
            f"Esperado {expected_obj:.6f}, obtenido {obj:.6f}"
        )

    # -----------------------------------------------------------------------
    # 8. Término negativo crece linealmente con nu
    # -----------------------------------------------------------------------

    def test_negative_term_linear_in_nu_scaling(self, laguna, vc):
        """obj(alpha·nu) - obj(0·nu) == -max_bins·alpha·nu.sum() para cualquier alpha."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(42)
        mu_fixed = np.random.rand(n_j, n_k)
        nu_base  = np.random.rand(n_j, n_k)
        mb = laguna.params.max_bins

        m_base = Multipliers(mu=mu_fixed, lbd=np.zeros(n_j), nu=np.zeros((n_j, n_k)))
        _, _, obj_base = solve_plr_yw(laguna, m_base, vc)

        for alpha in [0.5, 1.0, 2.0, 5.0]:
            m = Multipliers(mu=mu_fixed, lbd=np.zeros(n_j), nu=alpha * nu_base)
            _, _, obj = solve_plr_yw(laguna, m, vc)
            expected_delta = -mb * alpha * nu_base.sum()
            actual_delta   = obj - obj_base
            assert actual_delta == pytest.approx(expected_delta, rel=1e-9), (
                f"alpha={alpha}: delta={actual_delta:.6f} esperado={expected_delta:.6f}"
            )

    def test_negative_term_formula_nu_uniform(self, laguna, vc):
        """Con mu=0 y nu=c·ones: obj = -max_bins·c·n_j·n_k (fórmula cerrada)."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        mb = laguna.params.max_bins
        for c in [0.5, 1.0, 3.7]:
            m = Multipliers(
                mu=np.zeros((n_j, n_k)),
                lbd=np.zeros(n_j),
                nu=np.full((n_j, n_k), c),
            )
            _, _, obj = solve_plr_yw(laguna, m, vc)
            expected = -mb * c * n_j * n_k
            assert obj == pytest.approx(expected, rel=1e-9), (
                f"c={c}: obj={obj:.4f} esperado={expected:.4f}"
            )

    def test_positive_term_linear_in_mu_scaling(self, laguna, vc):
        """Escalando mu por alpha, obj crece en alpha·positive_term_base."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(13)
        mu_base = np.random.rand(n_j, n_k)
        m0 = Multipliers(mu=np.zeros((n_j, n_k)), lbd=np.zeros(n_j), nu=np.zeros((n_j, n_k)))
        _, _, obj0 = solve_plr_yw(laguna, m0, vc)   # = 0.0

        m1 = Multipliers(mu=mu_base, lbd=np.zeros(n_j), nu=np.zeros((n_j, n_k)))
        _, _, obj1 = solve_plr_yw(laguna, m1, vc)
        pos1 = obj1 - obj0   # = positive_term con mu_base

        for alpha in [2.0, 0.5, 3.0]:
            m = Multipliers(mu=alpha * mu_base, lbd=np.zeros(n_j), nu=np.zeros((n_j, n_k)))
            _, _, obj = solve_plr_yw(laguna, m, vc)
            assert obj == pytest.approx(alpha * pos1, rel=1e-9), (
                f"alpha={alpha}: obj={obj:.6f} esperado={alpha * pos1:.6f}"
            )

    # -----------------------------------------------------------------------
    # Descomposición analítica del objetivo
    # -----------------------------------------------------------------------

    def test_obj_equals_positive_minus_negative_term(self, laguna, vc):
        """obj = Σ mu[j,k]·q_ik - max_bins·Σ nu[j,k] — verificado manualmente."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(0)
        mu = np.random.rand(n_j, n_k) * 0.5
        nu = np.random.rand(n_j, n_k) * 0.1
        m  = Multipliers(mu=mu, lbd=np.zeros(n_j), nu=nu)
        ya, _, obj = solve_plr_yw(laguna, m, vc)

        positive = sum(
            mu[ya[i, k], k] * compute_demand(laguna.I[i].h_i, laguna.params, k)
            for i in range(laguna.n_buildings)
            for k in range(laguna.n_waste_types)
            if ya[i, k] != -1
        )
        negative = laguna.params.max_bins * nu.sum()
        assert obj == pytest.approx(positive - negative, rel=1e-9)

    # -----------------------------------------------------------------------
    # Tests de regresión (valores fijados sobre instancia real)
    # -----------------------------------------------------------------------

    def test_regression_obj_zero_multipliers(self, yw_zero):
        """Regresión: obj = 0.0 con todos los multiplicadores a cero."""
        _, _, obj = yw_zero
        assert obj == 0.0

    def test_regression_no_unassigned_buildings(self, laguna, yw_zero):
        """En la instancia Laguna, todos los (i,k) tienen candidato válido."""
        ya, _, _ = yw_zero
        n_minus1 = int(np.sum(ya == -1))
        assert n_minus1 == 0, (
            f"{n_minus1} pares (i,k) sin asignar — se esperaba 0"
        )

    def test_regression_obj_mu1_nu0(self, yw_mu1):
        """Con mu=1 y nu=0, obj = suma de todas las demandas asignadas."""
        _, _, obj = yw_mu1
        assert obj == pytest.approx(12805.596276884233, rel=1e-9)

    def test_regression_obj_nu1_mu0(self, laguna, vc):
        """Con mu=0 y nu=1, obj = -max_bins·n_j·n_k = -4256."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        m = Multipliers(
            mu=np.zeros((n_j, n_k)),
            lbd=np.zeros(n_j),
            nu=np.ones((n_j, n_k)),
        )
        _, _, obj = solve_plr_yw(laguna, m, vc)
        expected = -float(laguna.params.max_bins * n_j * n_k)
        assert obj == pytest.approx(expected, rel=1e-9), (
            f"Esperado {expected}, obtenido {obj}"
        )

    def test_regression_obj_random_seed0(self, laguna, vc):
        """Regresión con multiplicadores aleatorios (seed=0): valor fijado."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(0)
        mu = np.random.rand(n_j, n_k) * 0.5
        nu = np.random.rand(n_j, n_k) * 0.1
        m  = Multipliers(mu=mu, lbd=np.zeros(n_j), nu=nu)
        _, _, obj = solve_plr_yw(laguna, m, vc)
        assert obj == pytest.approx(3144.8001173912235, rel=1e-9)


# ===========================================================================
# 7. Tests de solve_plr_x
# ===========================================================================

@pytest.fixture(scope="module")
def total_demand(laguna) -> np.ndarray:
    """Demanda total Σᵢ q_ik por tipo k sobre la instancia Laguna."""
    n_k = laguna.n_waste_types
    td = np.zeros(n_k)
    for i in range(laguna.n_buildings):
        for k in range(n_k):
            td[k] += compute_demand(laguna.I[i].h_i, laguna.params, k)
    return td


@pytest.fixture(scope="module")
def plrx_zero(laguna, mults_zero) -> tuple[np.ndarray, float]:
    """solve_plr_x con todos los multiplicadores a cero."""
    return solve_plr_x(laguna, mults_zero)


class TestSolvePlrX:
    """
    Tests para solve_plr_x — subproblema de colocación de contenedores P_LR_x.

    Formulación:
        coef[j,k] = c_k + λ_j + ν_{jk} - μ_{jk} · Q_k
        x*        = greedy por coef ascendente cubriendo Σⱼ Q_k·x[j,k] ≥ Σᵢ q_ik
        obj_plrx  = Σ_{j,k} coef[j,k] · x[j,k]

    Propiedad estructural: el greedy asigna en un único paso todos los bins
    necesarios al primer candidato en orden (ceil de la demanda restante),
    lo que implica exactamente 1 candidato con bins por tipo de residuo.

    Instancia de referencia: instancia_laguna.json
      · 740 edificios, 133 candidatos, 4 tipos
      · bin_capacity = 120 para todos los tipos
      · bin_cost = {0: 350, 1: 300, 2: 250, 3: 500}
      · total_demand ≈ [6418.16, 1012.92, 4974.97, 399.53]
      · min_bins (mults=0) = [54, 9, 42, 4]
    """

    # -----------------------------------------------------------------------
    # 1. Shape y tipo de la salida
    # -----------------------------------------------------------------------

    def test_x_shape(self, laguna, plrx_zero):
        """x debe tener shape (n_candidates, n_waste_types)."""
        x, _ = plrx_zero
        assert x.shape == (laguna.n_candidates, laguna.n_waste_types), (
            f"Shape inesperado: {x.shape}"
        )

    def test_x_dtype_is_integer(self, plrx_zero):
        """x debe ser un array de enteros (número de contenedores)."""
        x, _ = plrx_zero
        assert np.issubdtype(x.dtype, np.integer), (
            f"x debe ser dtype entero, obtenido {x.dtype}"
        )

    def test_obj_is_scalar_float(self, plrx_zero):
        """obj_plrx debe ser un float escalar."""
        _, obj = plrx_zero
        assert isinstance(obj, float), f"obj_plrx debe ser float, obtenido {type(obj)}"
        assert np.ndim(obj) == 0

    # -----------------------------------------------------------------------
    # 2. Restricción (26): Σⱼ Q_k · x[j,k] ≥ Σᵢ q_ik para todo k
    # -----------------------------------------------------------------------

    def test_capacity_covers_demand_zero_multipliers(self, laguna, plrx_zero, total_demand):
        """Con mults=0, la capacidad instalada cubre la demanda para cada tipo k."""
        x, _ = plrx_zero
        bin_cap = np.array([laguna.params.bin_capacity[k] for k in laguna.K])
        for k in range(laguna.n_waste_types):
            installed = float(np.sum(x[:, k]) * bin_cap[k])
            assert installed >= total_demand[k] - 1e-9, (
                f"k={k}: capacidad={installed:.4f} < demanda={total_demand[k]:.4f}"
            )

    @pytest.mark.parametrize("seed", [42, 7, 99])
    def test_capacity_covers_demand_random_multipliers(self, laguna, total_demand, seed):
        """Para multiplicadores aleatorios, la restricción (26) se cumple siempre."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(seed)
        m = Multipliers(
            mu=np.random.rand(n_j, n_k) * 0.5,
            lbd=np.random.rand(n_j) * 0.1,
            nu=np.random.rand(n_j, n_k) * 0.1,
        )
        x, _ = solve_plr_x(laguna, m)
        bin_cap = np.array([laguna.params.bin_capacity[k] for k in laguna.K])
        for k in range(n_k):
            installed = float(np.sum(x[:, k]) * bin_cap[k])
            assert installed >= total_demand[k] - 1e-9, (
                f"seed={seed} k={k}: capacidad={installed:.4f} < demanda={total_demand[k]:.4f}"
            )

    # -----------------------------------------------------------------------
    # 3. Con multiplicadores a cero: coef[j,k] = c_k → bins mínimos en j=0
    # -----------------------------------------------------------------------

    def test_zero_multipliers_total_bins_is_minimum_per_type(
        self, laguna, plrx_zero, total_demand
    ):
        """Con mults=0, el total de bins instalados por tipo es el mínimo necesario."""
        import math
        x, _ = plrx_zero
        bin_cap = np.array([laguna.params.bin_capacity[k] for k in laguna.K])
        for k in range(laguna.n_waste_types):
            min_bins = math.ceil(total_demand[k] / bin_cap[k])
            actual = int(np.sum(x[:, k]))
            assert actual == min_bins, (
                f"k={k}: bins instalados={actual}, mínimo esperado={min_bins}"
            )

    def test_zero_multipliers_only_j0_has_bins(self, plrx_zero):
        """Con mults=0, coefs iguales por tipo: solo j=0 recibe bins (argsort ASC)."""
        x, _ = plrx_zero
        for k in range(x.shape[1]):
            nonzero = np.where(x[:, k] > 0)[0]
            assert len(nonzero) == 1, (
                f"k={k}: {len(nonzero)} candidatos con bins, esperado 1"
            )
            assert nonzero[0] == 0, (
                f"k={k}: bins en j={nonzero[0]}, esperado j=0"
            )

    def test_zero_multipliers_obj_equals_cost_times_min_bins(
        self, laguna, plrx_zero, total_demand
    ):
        """Con mults=0: obj = Σ_k c_k · ceil(demand_k / Q_k)."""
        import math
        _, obj = plrx_zero
        bin_cost = np.array([laguna.params.bin_cost[k] for k in laguna.K])
        bin_cap = np.array([laguna.params.bin_capacity[k] for k in laguna.K])
        expected = float(sum(
            bin_cost[k] * math.ceil(total_demand[k] / bin_cap[k])
            for k in range(laguna.n_waste_types)
        ))
        assert obj == pytest.approx(expected, rel=1e-9)

    # -----------------------------------------------------------------------
    # 4. Con mu muy grande en un candidato concreto, ese candidato recibe todos los bins
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("j_star", [0, 5, 50, 132])
    def test_large_mu_concentrates_all_bins_at_j_star(
        self, laguna, total_demand, j_star
    ):
        """
        mu[j*,:] = 1e6 → coef[j*,k] << 0 → greedy pone todos los bins en j*.
        Ningún otro candidato recibe bins.
        """
        import math
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        bin_cap = np.array([laguna.params.bin_capacity[k] for k in laguna.K])
        mu = np.zeros((n_j, n_k))
        mu[j_star, :] = 1e6
        m = Multipliers(mu=mu, lbd=np.zeros(n_j), nu=np.zeros((n_j, n_k)))
        x, _ = solve_plr_x(laguna, m)

        for k in range(n_k):
            min_bins = math.ceil(total_demand[k] / bin_cap[k])
            assert x[j_star, k] == min_bins, (
                f"j*={j_star} k={k}: x={x[j_star,k]}, esperado {min_bins}"
            )
            assert int(np.sum(x[:, k])) == min_bins, (
                f"j*={j_star} k={k}: total bins={np.sum(x[:,k])}, "
                f"esperado {min_bins} (concentrados en j*)"
            )

    # -----------------------------------------------------------------------
    # 5. obj_plrx = Σ_{j,k} coef[j,k] · x[j,k] — verificación directa
    # -----------------------------------------------------------------------

    def test_obj_equals_coef_dot_x_zero_multipliers(self, laguna, plrx_zero):
        """Con mults=0: obj = Σ_{j,k} c_k · x[j,k] (fórmula directa)."""
        x, obj = plrx_zero
        bin_cost = np.array([laguna.params.bin_cost[k] for k in laguna.K])
        expected = float(np.sum(bin_cost * x))
        assert obj == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("seed", [0, 42, 7])
    def test_obj_equals_coef_dot_x_random_multipliers(self, laguna, seed):
        """obj = Σ_{j,k} coef[j,k]·x[j,k] con multiplicadores aleatorios (seed={seed})."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(seed)
        mu = np.random.rand(n_j, n_k) * 0.5
        lbd = np.random.rand(n_j) * 0.1
        nu = np.random.rand(n_j, n_k) * 0.1
        m = Multipliers(mu=mu, lbd=lbd, nu=nu)
        x, obj = solve_plr_x(laguna, m)
        bin_cost = np.array([laguna.params.bin_cost[k] for k in laguna.K])
        bin_cap = np.array([laguna.params.bin_capacity[k] for k in laguna.K])
        coef = bin_cost + lbd[:, np.newaxis] + nu - mu * bin_cap
        expected = float(np.sum(coef * x))
        assert obj == pytest.approx(expected, rel=1e-9), (
            f"seed={seed}: obj={obj:.6f}, expected={expected:.6f}"
        )

    # -----------------------------------------------------------------------
    # 6. x contiene solo enteros no negativos
    # -----------------------------------------------------------------------

    def test_x_non_negative_zero_multipliers(self, plrx_zero):
        """Todos los elementos de x son ≥ 0 con multiplicadores cero."""
        x, _ = plrx_zero
        assert np.all(x >= 0), f"x tiene valores negativos: min={x.min()}"

    @pytest.mark.parametrize("seed", [13, 21, 55])
    def test_x_non_negative_random_multipliers(self, laguna, seed):
        """x ≥ 0 incluso con coeficientes muy negativos (mu grande)."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(seed)
        m = Multipliers(
            mu=np.random.rand(n_j, n_k) * 2.0,
            lbd=np.random.rand(n_j) * 0.1,
            nu=np.random.rand(n_j, n_k) * 0.05,
        )
        x, _ = solve_plr_x(laguna, m)
        assert np.all(x >= 0), f"seed={seed}: x tiene valores negativos, min={x.min()}"

    def test_x_integer_valued(self, plrx_zero):
        """x contiene exclusivamente enteros (no fracciones de contenedor)."""
        x, _ = plrx_zero
        assert np.array_equal(x, x.astype(int)), "x contiene valores no enteros"

    # -----------------------------------------------------------------------
    # 7. Caso límite: demanda total cero → x = 0 para ese tipo
    # -----------------------------------------------------------------------

    def test_zero_demand_yields_zero_x_and_zero_obj(self):
        """h_i=0 para todos los edificios → demanda=0 → x=0 y obj=0.0."""
        I = {0: _bld(0, h_i=0), 1: _bld(1, h_i=0)}
        J = {j: _cand(j) for j in range(3)}
        inst = _make_instance(I, J, dij={})
        m = Multipliers(
            mu=np.zeros((3, 4)),
            lbd=np.zeros(3),
            nu=np.zeros((3, 4)),
        )
        x, obj = solve_plr_x(inst, m)
        assert np.all(x == 0), f"Esperado x=0 con demanda nula, obtenido max={x.max()}"
        assert obj == 0.0, f"Esperado obj=0.0, obtenido {obj}"

    def test_zero_demand_yields_zero_x_with_nonzero_multipliers(self):
        """Con demanda cero, x=0 independientemente de los multiplicadores."""
        np.random.seed(11)
        I = {0: _bld(0, h_i=0)}
        J = {j: _cand(j) for j in range(3)}
        inst = _make_instance(I, J, dij={})
        m = Multipliers(
            mu=np.random.rand(3, 4),
            lbd=np.random.rand(3),
            nu=np.random.rand(3, 4),
        )
        x, _ = solve_plr_x(inst, m)
        assert np.all(x == 0), (
            "Con demanda=0 y mults≠0: x debe ser todo ceros"
        )

    def test_zero_demand_per_type_when_hi_zero_all_types(self):
        """h_i=0 anula la demanda de todos los tipos (q_ik = h_i · α · p_k = 0)."""
        I = {0: _bld(0, h_i=0)}
        J = {0: _cand(0)}
        inst = _make_instance(I, J, dij={})
        m = Multipliers(
            mu=np.zeros((1, 4)),
            lbd=np.zeros(1),
            nu=np.zeros((1, 4)),
        )
        x, _ = solve_plr_x(inst, m)
        for k in range(4):
            assert int(np.sum(x[:, k])) == 0, f"k={k}: x[:,k]={x[:,k]} debe ser ceros"

    # -----------------------------------------------------------------------
    # 8. Greedy instala en orden de coeficiente ascendente
    # -----------------------------------------------------------------------

    def test_greedy_lowest_coef_candidate_receives_all_bins(self):
        """
        Instancia sintética: 3 candidatos, 1 tipo, 1 bin necesario.
        lbd=[10, 5, 1] → coef=[310, 305, 301] → j=2 (coef mínimo) gana.
        j=0 y j=1 reciben 0 bins.
        """
        params = ModelParameters(
            opening_cost=4000.0, max_bins=8, nimby_distance=10.0,
            waste_per_capita=1.32, overflow_penalty=500.0,
            bin_cost={0: 300.0},
            bin_capacity={0: 100.0},
            coverage_radius={0: 150.0},
            waste_proportion={0: 1.0},
            collection_frequency={0: 1.0},
            lognormal_mu=0.0, lognormal_sigma=0.25, overflow_threshold=0.05,
        )
        # demand = 50 * 1.32 * 1.0 = 66 → ceil(66/100) = 1 bin
        I = {0: _bld(0, h_i=50)}
        J = {j: _cand(j) for j in range(3)}
        inst = _make_instance(I, J, dij={}, K=[0], params=params)

        lbd = np.array([10.0, 5.0, 1.0])   # coefs: 310, 305, 301
        m = Multipliers(mu=np.zeros((3, 1)), lbd=lbd, nu=np.zeros((3, 1)))
        x, obj = solve_plr_x(inst, m)

        assert x[2, 0] == 1, (
            f"j=2 (coef=301, mínimo) debe recibir 1 bin, obtenido {x[2,0]}"
        )
        assert x[0, 0] == 0, (
            f"j=0 (coef=310) no debe recibir bins, obtenido {x[0,0]}"
        )
        assert x[1, 0] == 0, (
            f"j=1 (coef=305) no debe recibir bins, obtenido {x[1,0]}"
        )
        assert obj == pytest.approx(301.0, rel=1e-9), f"obj={obj} esperado=301.0"

    def test_greedy_stops_once_demand_met(self):
        """
        2 candidatos, capacidad de bin > demanda total.
        El primero (coef menor) cubre la demanda en un único paso;
        el segundo no debe recibir ningún bin.
        """
        params = ModelParameters(
            opening_cost=4000.0, max_bins=8, nimby_distance=10.0,
            waste_per_capita=1.32, overflow_penalty=500.0,
            bin_cost={0: 300.0},
            bin_capacity={0: 200.0},  # 1 bin de 200 cubre demand=132
            coverage_radius={0: 150.0},
            waste_proportion={0: 1.0},
            collection_frequency={0: 1.0},
            lognormal_mu=0.0, lognormal_sigma=0.25, overflow_threshold=0.05,
        )
        # demand = 100 * 1.32 = 132 → ceil(132/200) = 1 bin
        I = {0: _bld(0, h_i=100)}
        J = {0: _cand(0), 1: _cand(1)}
        inst = _make_instance(I, J, dij={}, K=[0], params=params)

        lbd = np.array([0.0, 5.0])   # j=0 más barato → gana
        m = Multipliers(mu=np.zeros((2, 1)), lbd=lbd, nu=np.zeros((2, 1)))
        x, _ = solve_plr_x(inst, m)

        assert x[0, 0] == 1, f"j=0 debe recibir 1 bin, obtenido {x[0,0]}"
        assert x[1, 0] == 0, (
            f"j=1 no debe recibir bins (demanda cubierta tras j=0), obtenido {x[1,0]}"
        )

    def test_greedy_winner_has_minimum_coef_random_multipliers(self, laguna):
        """
        El candidato que recibe bins para cada tipo k debe ser aquel con
        coef[j,k] mínimo — verificado con multiplicadores aleatorios.
        """
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(42)
        mu = np.random.rand(n_j, n_k) * 0.5
        lbd = np.random.rand(n_j) * 0.1
        nu = np.random.rand(n_j, n_k) * 0.1
        m = Multipliers(mu=mu, lbd=lbd, nu=nu)
        x, _ = solve_plr_x(laguna, m)

        bin_cost = np.array([laguna.params.bin_cost[k] for k in laguna.K])
        bin_cap = np.array([laguna.params.bin_capacity[k] for k in laguna.K])
        coef = bin_cost + lbd[:, np.newaxis] + nu - mu * bin_cap

        for k in range(n_k):
            j_winner = int(np.where(x[:, k] > 0)[0][0])
            j_min = int(np.argmin(coef[:, k]))
            assert j_winner == j_min, (
                f"k={k}: j_winner={j_winner} (coef={coef[j_winner,k]:.4f}) "
                f"≠ j_argmin={j_min} (coef={coef[j_min,k]:.4f})"
            )

    def test_greedy_exactly_one_candidate_per_type(self, laguna):
        """
        Propiedad estructural: el greedy cubre toda la demanda en un único paso
        (ceil), por lo que exactamente 1 candidato por tipo tiene bins > 0.
        """
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(7)
        m = Multipliers(
            mu=np.random.rand(n_j, n_k) * 0.3,
            lbd=np.random.rand(n_j) * 0.1,
            nu=np.random.rand(n_j, n_k) * 0.05,
        )
        x, _ = solve_plr_x(laguna, m)
        for k in range(n_k):
            n_nonzero = int(np.sum(x[:, k] > 0))
            assert n_nonzero == 1, (
                f"k={k}: {n_nonzero} candidatos con bins, esperado exactamente 1"
            )

    # -----------------------------------------------------------------------
    # 9. Regresión (valores fijados sobre instancia real)
    # -----------------------------------------------------------------------

    def test_regression_obj_zero_multipliers(self, plrx_zero):
        """Regresión: obj = 34 100.0 con todos los multiplicadores a cero."""
        _, obj = plrx_zero
        assert obj == pytest.approx(34100.0, rel=1e-9)

    def test_regression_x_j0_zero_multipliers(self, plrx_zero):
        """Regresión: x[0,:] = [54, 9, 42, 4] (bins mínimos concentrados en j=0)."""
        x, _ = plrx_zero
        expected = np.array([54, 9, 42, 4])
        assert np.array_equal(x[0, :], expected), (
            f"x[0,:] = {x[0,:]} esperado {expected}"
        )

    def test_regression_obj_random_seed42(self, laguna):
        """Regresión: obj con multiplicadores aleatorios (seed=42, escala 0.5/0.1)."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(42)
        m = Multipliers(
            mu=np.random.rand(n_j, n_k) * 0.5,
            lbd=np.random.rand(n_j) * 0.1,
            nu=np.random.rand(n_j, n_k) * 0.1,
        )
        _, obj = solve_plr_x(laguna, m)
        assert obj == pytest.approx(27717.650706954864, rel=1e-9)

    def test_regression_obj_random_seed7(self, laguna):
        """Regresión: obj con multiplicadores aleatorios (seed=7, escala 0.3/0.1/0.05)."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(7)
        m = Multipliers(
            mu=np.random.rand(n_j, n_k) * 0.3,
            lbd=np.random.rand(n_j) * 0.1,
            nu=np.random.rand(n_j, n_k) * 0.05,
        )
        _, obj = solve_plr_x(laguna, m)
        assert obj == pytest.approx(30218.736234042663, rel=1e-9)


# ===========================================================================
# 8. Tests de solve_plr_z
# ===========================================================================

@pytest.fixture(scope="module")
def plrz_zero(laguna, mults_zero, vc) -> tuple[np.ndarray, float]:
    """solve_plr_z con todos los multiplicadores a cero."""
    return solve_plr_z(laguna, mults_zero, vc)


class TestSolvePlrZ:
    """
    Tests para solve_plr_z — subproblema de apertura de puntos P_LR_z.

    Formulación:
        coef[j] = C_j - N_j · λ_j
        z*  = arg min  Σ_j coef[j] · z_j
              s.t.   Σ_{j ∈ vc[i][k]} z_j ≥ 1   ∀ (i,k) con vc[i][k] ≠ ∅
                     z_j ∈ {0,1}
        obj_plrz = Σ_j coef[j] · z[j]

    Con multiplicadores a cero: coef[j] = C_j = 4000 ∀j → Gurobi minimiza
    el número de candidatos abiertos (minimum set cover).

    Instancia de referencia: instancia_laguna.json
      · 740 edificios, 133 candidatos, 4 tipos de residuo
      · opening_cost = 4000, max_bins = 8
      · Mínimo set cover (mults=0) = 27 candidatos → obj = 108 000
      · Candidatos forzados (únicos para algún (i,k)): j=51, j=73
    """

    # -----------------------------------------------------------------------
    # 1. Shape correcta de z — (n_candidates,) dtype bool
    # -----------------------------------------------------------------------

    def test_z_shape(self, laguna, plrz_zero):
        """z debe tener shape (n_candidates,)."""
        z, _ = plrz_zero
        assert z.shape == (laguna.n_candidates,), (
            f"Shape inesperado: {z.shape}, esperado ({laguna.n_candidates},)"
        )

    def test_z_dtype_is_bool(self, plrz_zero):
        """z debe ser de tipo bool (0/1 binario)."""
        z, _ = plrz_zero
        assert z.dtype == bool, f"z debe ser dtype=bool, obtenido {z.dtype}"

    def test_obj_is_scalar_float(self, plrz_zero):
        """obj_plrz debe ser un float escalar."""
        _, obj = plrz_zero
        assert isinstance(obj, float), f"obj_plrz debe ser float, obtenido {type(obj)}"
        assert np.ndim(obj) == 0

    # -----------------------------------------------------------------------
    # 2. Restricción (23): cobertura completa de todos los (i,k) factibles
    # -----------------------------------------------------------------------

    def test_coverage_constraint_zero_multipliers(self, laguna, vc, plrz_zero):
        """Para todo (i,k) con vc[i][k] ≠ ∅, any(z[j] for j in vc[i][k]) es True."""
        z, _ = plrz_zero
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                if vc[i][k]:
                    assert any(z[j] for j in vc[i][k]), (
                        f"Restricción (23) violada: (i={i},k={k}) sin candidato abierto, "
                        f"vc={vc[i][k]}, z_vc={[bool(z[j]) for j in vc[i][k]]}"
                    )

    @pytest.mark.parametrize("seed", [42, 7, 99])
    def test_coverage_constraint_random_multipliers(self, laguna, vc, seed):
        """Restricción (23) se satisface para cualquier configuración de multiplicadores."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(seed)
        lbd = np.random.rand(n_j) * 0.4   # coef[j] = 4000 - lbd*8 > 0 siempre
        m = Multipliers(mu=np.zeros((n_j, n_k)), lbd=lbd, nu=np.zeros((n_j, n_k)))
        z, _ = solve_plr_z(laguna, m, vc)
        for i in range(laguna.n_buildings):
            for k in range(n_k):
                if vc[i][k]:
                    assert any(z[j] for j in vc[i][k]), (
                        f"seed={seed}: (i={i},k={k}) sin candidato abierto, "
                        f"vc={vc[i][k]}"
                    )

    # -----------------------------------------------------------------------
    # 3. Con mults=0, coef[j]=C_j ∀j → abre el mínimo número de candidatos
    # -----------------------------------------------------------------------

    def test_zero_multipliers_coefs_all_equal_opening_cost(self, laguna, mults_zero):
        """coef[j] = opening_cost - 0·max_bins = opening_cost = 4000 para todo j."""
        coef = laguna.params.opening_cost - mults_zero.lbd * laguna.params.max_bins
        assert np.all(coef == laguna.params.opening_cost), (
            f"Con lbd=0, todos los coefs deben ser {laguna.params.opening_cost}, "
            f"min={coef.min():.2f}, max={coef.max():.2f}"
        )

    def test_zero_multipliers_minimum_candidates_opened(self, plrz_zero):
        """Con coefs uniformes positivos, Gurobi resuelve el minimum set cover = 27."""
        z, _ = plrz_zero
        n_open = int(np.sum(z))
        assert n_open == 27, (
            f"Se esperaban 27 candidatos abiertos (mínimo set cover), obtenidos {n_open}"
        )

    def test_zero_multipliers_obj_equals_opening_cost_times_n_open(
        self, laguna, plrz_zero
    ):
        """obj = C_j · n_open = 4000 · 27 = 108 000 con mults=0."""
        z, obj = plrz_zero
        n_open = int(np.sum(z))
        expected = laguna.params.opening_cost * n_open
        assert obj == pytest.approx(expected, rel=1e-9), (
            f"Esperado {expected}, obtenido {obj}"
        )

    # -----------------------------------------------------------------------
    # 4. obj_plrz = Σ_j coef[j] · z[j] — verificación manual del coeficiente
    # -----------------------------------------------------------------------

    def test_obj_equals_coef_dot_z_zero_multipliers(self, laguna, plrz_zero):
        """Con mults=0: coef[j] = opening_cost → obj = opening_cost · Σ z[j]."""
        z, obj = plrz_zero
        coef = laguna.params.opening_cost * np.ones(laguna.n_candidates)
        expected = float(np.sum(coef * z))
        assert obj == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("seed", [42, 7])
    def test_obj_equals_coef_dot_z_random_multipliers(self, laguna, vc, seed):
        """obj = Σ_j (C_j - N_j·λ_j)·z[j] verificado manualmente (seed={seed})."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(seed)
        lbd = np.random.rand(n_j) * 0.5
        m = Multipliers(mu=np.zeros((n_j, n_k)), lbd=lbd, nu=np.zeros((n_j, n_k)))
        z, obj = solve_plr_z(laguna, m, vc)
        coef = laguna.params.opening_cost - lbd * laguna.params.max_bins
        expected = float(np.sum(coef * z))
        assert obj == pytest.approx(expected, rel=1e-9), (
            f"seed={seed}: obj={obj:.6f}, recalculado={expected:.6f}"
        )

    # -----------------------------------------------------------------------
    # 5. Con λ_j* muy grande, coef[j*] < 0 → z[j*] = True
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("j_star", [0, 5, 50, 132])
    def test_negative_coef_candidate_always_open(self, laguna, vc, j_star):
        """λ[j*] = 501 → coef[j*] = 4000 - 501·8 = -8 < 0 → z[j*] debe ser True."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        lbd = np.zeros(n_j)
        lbd[j_star] = 501.0   # coef = 4000 - 501*8 = -8 < 0
        m = Multipliers(mu=np.zeros((n_j, n_k)), lbd=lbd, nu=np.zeros((n_j, n_k)))
        z, _ = solve_plr_z(laguna, m, vc)
        coef_jstar = laguna.params.opening_cost - lbd[j_star] * laguna.params.max_bins
        assert coef_jstar < 0, (
            f"Precondición: coef[{j_star}]={coef_jstar} debe ser negativo"
        )
        assert z[j_star] == True, (
            f"j*={j_star} con coef={coef_jstar:.1f} < 0 debe estar abierto, "
            f"z[j*]={z[j_star]}"
        )

    def test_all_negative_coefs_all_candidates_open(self, laguna, vc):
        """Con λ_j = 501 ∀j (coef[j] = -8 ∀j), todos los 133 candidatos deben abrirse."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        lbd = np.full(n_j, 501.0)
        m = Multipliers(mu=np.zeros((n_j, n_k)), lbd=lbd, nu=np.zeros((n_j, n_k)))
        z, _ = solve_plr_z(laguna, m, vc)
        assert np.all(z), (
            f"Con todos los coefs negativos deben abrirse todos los candidatos; "
            f"cerrados: {np.where(~z)[0].tolist()}"
        )

    # -----------------------------------------------------------------------
    # 6. Sin candidatos abiertos "inútiles" (con coefs positivos)
    # -----------------------------------------------------------------------

    def test_no_useless_open_candidates_zero_multipliers(self, laguna, vc, plrz_zero):
        """Con mults=0 (coefs > 0), todo j abierto cubre al menos un par (i,k)."""
        z, _ = plrz_zero
        n_k = laguna.n_waste_types
        for j in range(laguna.n_candidates):
            if z[j]:
                covers = any(
                    j in vc[i][k]
                    for i in range(laguna.n_buildings)
                    for k in range(n_k)
                )
                assert covers, (
                    f"j={j} está abierto pero no cubre ningún par (i,k) — candidato inútil"
                )

    def test_every_open_candidate_uniquely_necessary(self, laguna, vc, plrz_zero):
        """
        Con coefs uniformes > 0 (minimum set cover), todo j abierto cubre al menos
        un par (i,k) que ningún otro candidato abierto cubre.
        Invariante de optimalidad: si j fuera redundante, retirarlo reduciría el coste.
        """
        z, _ = plrz_zero
        open_set = set(int(j) for j in np.where(z)[0])
        n_k = laguna.n_waste_types
        for j in open_set:
            uniquely_covers = any(
                j in vc[i][k] and open_set.isdisjoint(set(vc[i][k]) - {j})
                for i in range(laguna.n_buildings)
                for k in range(n_k)
            )
            assert uniquely_covers, (
                f"j={j} es redundante en el minimum set cover — "
                f"ningún (i,k) depende exclusivamente de él"
            )

    # -----------------------------------------------------------------------
    # 7. Número de candidatos abiertos razonable: 0 < n_open < n_candidates
    # -----------------------------------------------------------------------

    def test_n_open_positive_zero_multipliers(self, plrz_zero):
        """Debe abrirse al menos 1 candidato."""
        z, _ = plrz_zero
        assert int(np.sum(z)) > 0, "n_open debe ser > 0"

    def test_n_open_less_than_total_zero_multipliers(self, laguna, plrz_zero):
        """Con coefs positivos, no es óptimo abrir todos los candidatos."""
        z, _ = plrz_zero
        n_open = int(np.sum(z))
        assert n_open < laguna.n_candidates, (
            f"No es razonable abrir todos los candidatos con coefs > 0: "
            f"{n_open}/{laguna.n_candidates}"
        )

    @pytest.mark.parametrize("seed", [42, 7, 99])
    def test_n_open_reasonable_random_positive_multipliers(self, laguna, vc, seed):
        """Con lbd < 500 (coefs > 0), n_open está estrictamente entre 0 y n_j."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(seed)
        lbd = np.random.rand(n_j) * 0.4   # coef min = 4000 - 0.4*8 = 3996.8 > 0
        m = Multipliers(mu=np.zeros((n_j, n_k)), lbd=lbd, nu=np.zeros((n_j, n_k)))
        z, _ = solve_plr_z(laguna, m, vc)
        n_open = int(np.sum(z))
        assert 0 < n_open < n_j, (
            f"seed={seed}: n_open={n_open} debe estar en (0, {n_j})"
        )

    # -----------------------------------------------------------------------
    # 8. Candidatos forzados (único válido para algún (i,k)) siempre abiertos
    # -----------------------------------------------------------------------

    def test_forced_candidates_open_zero_multipliers(self, laguna, vc, plrz_zero):
        """j con |vc[i][k]|=1 para algún (i,k) debe aparecer en z — es ineludible."""
        z, _ = plrz_zero
        forced = {
            j
            for i in range(laguna.n_buildings)
            for k in range(laguna.n_waste_types)
            for j in vc[i][k]
            if len(vc[i][k]) == 1
        }
        for j in forced:
            assert z[j], (
                f"j={j} es el único candidato para algún (i,k) pero no está abierto"
            )

    @pytest.mark.parametrize("j_forced,i_forced,k_forced", [
        (51, 172, 0),   # j=51 único para (i=172, k=0) orgánico
        (73, 362, 0),   # j=73 único para (i=362, k=0) orgánico
    ])
    def test_specific_forced_candidate_open(
        self, laguna, vc, plrz_zero, j_forced, i_forced, k_forced
    ):
        """Regresión: j=51 y j=73 son los únicos candidatos para ciertos edificios."""
        z, _ = plrz_zero
        assert len(vc[i_forced][k_forced]) == 1, (
            f"Precondición: (i={i_forced},k={k_forced}) debe tener exactamente 1 candidato"
        )
        assert vc[i_forced][k_forced][0] == j_forced, (
            f"Precondición: único candidato de (i={i_forced},k={k_forced}) debe ser j={j_forced}"
        )
        assert z[j_forced], (
            f"j={j_forced} único para (i={i_forced},k={k_forced}) → debe estar abierto"
        )

    @pytest.mark.parametrize("seed", [0, 42, 7])
    def test_forced_candidates_open_with_random_multipliers(self, laguna, vc, seed):
        """j con única cobertura está abierto independientemente de los multiplicadores."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(seed)
        lbd = np.random.rand(n_j) * 0.4
        m = Multipliers(mu=np.zeros((n_j, n_k)), lbd=lbd, nu=np.zeros((n_j, n_k)))
        z, _ = solve_plr_z(laguna, m, vc)
        forced = {
            j
            for i in range(laguna.n_buildings)
            for k in range(n_k)
            for j in vc[i][k]
            if len(vc[i][k]) == 1
        }
        for j in forced:
            assert z[j], (
                f"seed={seed}: candidato forzado j={j} no está abierto"
            )

    # -----------------------------------------------------------------------
    # 9. Regresión (valores fijados sobre instancia_laguna.json)
    # -----------------------------------------------------------------------

    def test_regression_obj_zero_multipliers(self, plrz_zero):
        """Regresión: obj = 108 000.0 con todos los multiplicadores a cero."""
        _, obj = plrz_zero
        assert obj == pytest.approx(108_000.0, rel=1e-9)

    def test_regression_n_open_zero_multipliers(self, plrz_zero):
        """Regresión: minimum set cover sobre instancia Laguna = 27 candidatos."""
        z, _ = plrz_zero
        assert int(np.sum(z)) == 27

    def test_regression_obj_seed42(self, laguna, vc):
        """Regresión: obj con lbd ~ U(0, 0.5) (seed=42)."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(42)
        lbd = np.random.rand(n_j) * 0.5
        m = Multipliers(mu=np.zeros((n_j, n_k)), lbd=lbd, nu=np.zeros((n_j, n_k)))
        _, obj = solve_plr_z(laguna, m, vc)
        assert obj == pytest.approx(107_925.29823913104, rel=1e-9)

    def test_regression_obj_seed7(self, laguna, vc):
        """Regresión: obj con lbd ~ U(0, 0.3) (seed=7)."""
        n_j, n_k = laguna.n_candidates, laguna.n_waste_types
        np.random.seed(7)
        lbd = np.random.rand(n_j) * 0.3
        m = Multipliers(mu=np.zeros((n_j, n_k)), lbd=lbd, nu=np.zeros((n_j, n_k)))
        _, obj = solve_plr_z(laguna, m, vc)
        assert obj == pytest.approx(107_961.94154481686, rel=1e-9)


# ===========================================================================
# 9. Tests de repair_solution
# ===========================================================================

@pytest.fixture(scope="module")
def z_from_plrz(plrz_zero) -> np.ndarray:
    """z del minimum set cover (mults=0, 27 candidatos) — input para repair_solution."""
    z, _ = plrz_zero
    return z


@pytest.fixture(scope="module")
def repair_zero(laguna, z_from_plrz, vc) -> FeasibleSolution:
    """
    FeasibleSolution de repair_solution sobre z del minimum set cover.

    Nota: el set cover mínimo abre solo 27 candidatos para 740 edificios y
    4 tipos de residuo. Esto provoca que 7 candidatos excedan max_bins=8;
    repair_solution emite UserWarning pero no restringe x. Los tests de
    constraint (5) lo verifican explícitamente.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return repair_solution(laguna, z_from_plrz, vc)


@pytest.fixture(scope="module")
def repair_all_open(laguna, vc) -> FeasibleSolution:
    """FeasibleSolution de repair_solution con todos los candidatos abiertos."""
    z_all = np.ones(laguna.n_candidates, dtype=bool)
    return repair_solution(laguna, z_all, vc)


class TestRepairSolution:
    """
    Tests para repair_solution.

    Invariantes del HDM que repair_solution debe garantizar:
      (1)  Shapes y tipos correctos de z, x, y_assign, w, cost.
      (2)  y_assign solo contiene índices de candidatos abiertos o -1 — nunca un
           candidato cerrado.
      (3)  Nearest-allocation: j asignado es el primero abierto en vc[i][k]
           (verificado contra dij y valid_candidates).
      (4)  Cobertura completa sobre la instancia Laguna (ningún y_assign[i,k]==-1).
      (5)  Restricción capacidad (4): demanda asignada ≤ Q_k·x[j,k] para todo (j,k).
      (6)  Restricción física (5): Σ_k x[j,k] ≤ max_bins, verificado con z_all;
           cuando se viola (z escaso), repair_solution emite UserWarning.
      (7)  w consistente con x: w[j,k] == (x[j,k] > 0) para todo (j,k).
      (8)  Coste correcto: cost = Σ_j opening_cost·z[j] + Σ_{j,k} bin_cost[k]·x[j,k].
      (9)  x[j,k] == 0 para todo j con z[j]=False — no hay contenedores en puntos
           cerrados.
      (10) Con z=all-True, y_assign[i,k] == vc[i][k][0] (el más cercano absoluto).

    Instancia de referencia: instancia_laguna.json (740 edif., 133 cand., 4 tipos)
      · z del minimum set cover (mults=0): 27 candidatos, cost=168 000 €
      · 7 candidatos superan max_bins=8 con ese z (j=6,10,17,19,24,28,56)
      · Con z=all-True: ningún candidato supera max_bins, cost=683 900 €
    """

    # -----------------------------------------------------------------------
    # Helper estático
    # -----------------------------------------------------------------------

    @staticmethod
    def _demand_at(instance, y_assign: np.ndarray) -> np.ndarray:
        """Recalcula demand_at[j,k] desde y_assign para verificar invariantes."""
        n_j = len(instance.J)
        n_k = len(instance.K)
        da = np.zeros((n_j, n_k))
        for i in range(len(instance.I)):
            for k in range(n_k):
                j = y_assign[i, k]
                if j != -1:
                    da[j, k] += compute_demand(instance.I[i].h_i, instance.params, k)
        return da

    # -----------------------------------------------------------------------
    # 1. Shapes y tipos correctos
    # -----------------------------------------------------------------------

    def test_z_shape(self, laguna, repair_zero):
        """z debe tener shape (n_candidates,)."""
        assert repair_zero.z.shape == (laguna.n_candidates,), (
            f"z shape: esperado ({laguna.n_candidates},), obtenido {repair_zero.z.shape}"
        )

    def test_x_shape(self, laguna, repair_zero):
        """x debe tener shape (n_candidates, n_waste_types)."""
        assert repair_zero.x.shape == (laguna.n_candidates, laguna.n_waste_types), (
            f"x shape: esperado ({laguna.n_candidates},{laguna.n_waste_types}), "
            f"obtenido {repair_zero.x.shape}"
        )

    def test_y_assign_shape(self, laguna, repair_zero):
        """y_assign debe tener shape (n_buildings, n_waste_types)."""
        assert repair_zero.y_assign.shape == (laguna.n_buildings, laguna.n_waste_types), (
            f"y_assign shape: esperado ({laguna.n_buildings},{laguna.n_waste_types}), "
            f"obtenido {repair_zero.y_assign.shape}"
        )

    def test_w_shape(self, laguna, repair_zero):
        """w debe tener shape (n_candidates, n_waste_types)."""
        assert repair_zero.w.shape == (laguna.n_candidates, laguna.n_waste_types), (
            f"w shape: esperado ({laguna.n_candidates},{laguna.n_waste_types}), "
            f"obtenido {repair_zero.w.shape}"
        )

    def test_z_dtype_bool(self, repair_zero):
        """z debe ser dtype=bool."""
        assert repair_zero.z.dtype == bool, (
            f"z debe ser bool, obtenido {repair_zero.z.dtype}"
        )

    def test_x_dtype_integer(self, repair_zero):
        """x debe ser dtype entero (número de contenedores)."""
        assert np.issubdtype(repair_zero.x.dtype, np.integer), (
            f"x debe ser entero, obtenido {repair_zero.x.dtype}"
        )

    def test_y_assign_dtype_integer(self, repair_zero):
        """y_assign debe ser dtype entero (índice de candidato o -1)."""
        assert np.issubdtype(repair_zero.y_assign.dtype, np.integer), (
            f"y_assign debe ser entero, obtenido {repair_zero.y_assign.dtype}"
        )

    def test_w_dtype_bool(self, repair_zero):
        """w debe ser dtype=bool."""
        assert repair_zero.w.dtype == bool, (
            f"w debe ser bool, obtenido {repair_zero.w.dtype}"
        )

    def test_cost_is_float(self, repair_zero):
        """cost debe ser un float (o compatible)."""
        assert isinstance(repair_zero.cost, (float, np.floating)), (
            f"cost debe ser float, obtenido {type(repair_zero.cost)}"
        )

    def test_x_non_negative(self, repair_zero):
        """x ≥ 0 para todo (j,k) — no hay contenedores negativos."""
        assert np.all(repair_zero.x >= 0), (
            f"x tiene valores negativos: min={repair_zero.x.min()}"
        )

    # -----------------------------------------------------------------------
    # 2. y_assign solo contiene candidatos abiertos o -1
    # -----------------------------------------------------------------------

    def test_y_assign_only_open_or_minus1(self, repair_zero):
        """y_assign[i,k] debe ser -1 o un j con z[j]=True."""
        ya = repair_zero.y_assign
        z = repair_zero.z
        n_i, n_k = ya.shape
        for i in range(n_i):
            for k in range(n_k):
                j = int(ya[i, k])
                if j != -1:
                    assert z[j], (
                        f"y_assign[{i},{k}]={j} apunta a candidato cerrado (z[{j}]=False)"
                    )

    def test_y_assign_never_points_to_closed_candidate_vectorized(self, repair_zero):
        """Verificación vectorizada: ningún y_assign apunta a un candidato cerrado."""
        ya = repair_zero.y_assign
        z = repair_zero.z
        assigned = ya[ya != -1].astype(int)
        closed_set = set(int(j) for j in np.where(~z)[0])
        violations = [int(j) for j in assigned if j in closed_set]
        assert not violations, (
            f"y_assign apunta a {len(violations)} candidatos cerrados "
            f"(muestra): {violations[:5]}"
        )

    def test_y_assign_valid_range(self, laguna, repair_zero):
        """Índices en y_assign ∈ [0, n_candidates) o -1."""
        ya = repair_zero.y_assign
        n_j = laguna.n_candidates
        assigned = ya[ya != -1]
        assert np.all(assigned >= 0), "y_assign tiene valores negativos distintos de -1"
        assert np.all(assigned < n_j), (
            f"y_assign tiene índices ≥ n_candidates={n_j}: max={assigned.max()}"
        )

    # -----------------------------------------------------------------------
    # 3. Nearest-allocation: j asignado es el más cercano ABIERTO
    # -----------------------------------------------------------------------

    def test_nearest_allocation_matches_first_open_in_vc(self, laguna, vc, repair_zero):
        """y_assign[i,k] == primer j en vc[i][k] con z[j]=True."""
        ya = repair_zero.y_assign
        z = repair_zero.z
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                expected = -1
                for j in vc[i][k]:
                    if z[j]:
                        expected = j
                        break
                assert int(ya[i, k]) == expected, (
                    f"(i={i},k={k}): y_assign={ya[i,k]}, primer abierto en vc={expected}"
                )

    def test_nearest_allocation_distance_is_minimum_among_open(
        self, laguna, vc, repair_zero
    ):
        """
        La distancia al j asignado es ≤ a la de cualquier otro candidato abierto
        en vc[i][k] — nearest-allocation exhaustiva verificada contra dij.
        """
        ya = repair_zero.y_assign
        z = repair_zero.z
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                j = int(ya[i, k])
                if j == -1:
                    continue
                d_assigned = laguna.dij[j][i]
                for j2 in vc[i][k]:
                    if z[j2] and j2 != j:
                        d2 = laguna.dij[j2][i]
                        assert d_assigned <= d2 + 1e-9, (
                            f"(i={i},k={k}): asignado j={j} d={d_assigned:.4f} > "
                            f"abierto j2={j2} d2={d2:.4f} — no es el más cercano"
                        )

    def test_nearest_allocation_candidate_is_valid_for_ik(
        self, laguna, vc, repair_zero
    ):
        """El j asignado siempre pertenece a valid_candidates[i][k]."""
        ya = repair_zero.y_assign
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                j = int(ya[i, k])
                if j == -1:
                    continue
                vc_set = set(vc[i][k])
                assert j in vc_set, (
                    f"(i={i},k={k}): j={j} asignado no está en valid_candidates"
                )

    def test_y_minus1_iff_no_open_candidate_in_vc(self, laguna, vc, repair_zero):
        """y_assign[i,k]==-1 ↔ no hay ningún candidato abierto en vc[i][k]."""
        ya = repair_zero.y_assign
        z = repair_zero.z
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                has_open = any(z[j] for j in vc[i][k])
                is_minus1 = (int(ya[i, k]) == -1)
                assert is_minus1 != has_open, (
                    f"(i={i},k={k}): y=-1={is_minus1} pero has_open={has_open} — inconsistente"
                )

    # -----------------------------------------------------------------------
    # 4. Cobertura completa (ningún -1 en la instancia Laguna)
    # -----------------------------------------------------------------------

    def test_full_coverage_no_minus1(self, repair_zero):
        """En la instancia Laguna con z=minimum set cover, cobertura al 100%."""
        n_minus1 = int(np.sum(repair_zero.y_assign == -1))
        assert n_minus1 == 0, (
            f"{n_minus1} pares (i,k) sin asignación — se esperaba cobertura completa"
        )

    def test_full_coverage_per_waste_type(self, laguna, repair_zero):
        """Ningún tipo de residuo tiene edificios sin asignación."""
        ya = repair_zero.y_assign
        for k in range(laguna.n_waste_types):
            unassigned = int(np.sum(ya[:, k] == -1))
            assert unassigned == 0, (
                f"k={k}: {unassigned} edificios sin asignación"
            )

    def test_full_coverage_all_buildings_have_j_for_all_k(self, laguna, repair_zero):
        """Cada edificio tiene asignación para los 4 tipos de residuo."""
        ya = repair_zero.y_assign
        for i in range(laguna.n_buildings):
            unassigned = int(np.sum(ya[i, :] == -1))
            assert unassigned == 0, (
                f"edificio {i}: {unassigned} tipos sin asignación"
            )

    # -----------------------------------------------------------------------
    # 5. Restricción (4) capacidad: demanda ≤ Q_k · x[j,k]
    # -----------------------------------------------------------------------

    def test_capacity_constraint_per_candidate_and_type(self, laguna, repair_zero):
        """Para todo (j,k), la demanda asignada no supera Q_k·x[j,k]."""
        demand_at = self._demand_at(laguna, repair_zero.y_assign)
        x = repair_zero.x
        bin_cap = np.array([laguna.params.bin_capacity[k] for k in laguna.K])
        for j in range(laguna.n_candidates):
            for k in range(laguna.n_waste_types):
                cap = float(x[j, k]) * bin_cap[k]
                dem = demand_at[j, k]
                assert dem <= cap + 1e-9, (
                    f"j={j} k={k}: demanda={dem:.4f} > capacidad={cap:.4f} "
                    f"(x[j,k]={x[j,k]}, Q_k={bin_cap[k]})"
                )

    def test_capacity_constraint_vectorized(self, laguna, repair_zero):
        """Vectorizado: demand_at ≤ x·Q_k para todo (j,k) — sin excepciones."""
        demand_at = self._demand_at(laguna, repair_zero.y_assign)
        x = repair_zero.x
        bin_cap = np.array([laguna.params.bin_capacity[k] for k in laguna.K])
        surplus = x * bin_cap - demand_at
        violations = list(zip(*np.where(surplus < -1e-9)))
        assert not violations, (
            f"{len(violations)} pares (j,k) violan la capacidad: "
            f"primeros 5 = {violations[:5]}"
        )

    def test_x_equals_ceil_demand_over_capacity(self, laguna, repair_zero):
        """x[j,k] == ceil(demand_at[j,k] / Q_k) cuando demand_at[j,k] > 0."""
        import math
        demand_at = self._demand_at(laguna, repair_zero.y_assign)
        x = repair_zero.x
        bin_cap = np.array([laguna.params.bin_capacity[k] for k in laguna.K])
        for j in range(laguna.n_candidates):
            for k in range(laguna.n_waste_types):
                if demand_at[j, k] > 0:
                    expected = math.ceil(demand_at[j, k] / bin_cap[k])
                    assert int(x[j, k]) == expected, (
                        f"x[{j},{k}]={x[j,k]}, ceil(demand/Q)={expected}, "
                        f"demand={demand_at[j,k]:.6f}, Q={bin_cap[k]}"
                    )

    def test_x_zero_where_no_demand(self, laguna, repair_zero):
        """x[j,k] == 0 cuando ningún edificio está asignado a (j,k)."""
        demand_at = self._demand_at(laguna, repair_zero.y_assign)
        x = repair_zero.x
        no_demand_mask = demand_at == 0.0
        assert np.all(x[no_demand_mask] == 0), (
            f"x>0 donde demand=0 en {int(np.sum(x[no_demand_mask] > 0))} posiciones"
        )

    # -----------------------------------------------------------------------
    # 6. Restricción (5) límite físico: Σ_k x[j,k] ≤ max_bins
    #    Con z escaso (27 cand.), repair_solution emite UserWarning para los
    #    candidatos que se sobrecargan. Con z=all-True (133 cand.), se satisface.
    # -----------------------------------------------------------------------

    def test_physical_limit_violated_emits_user_warning(
        self, laguna, z_from_plrz, vc
    ):
        """
        Con z del minimum set cover (27 candidatos), 7 de ellos superan max_bins=8;
        repair_solution debe emitir UserWarning por cada uno de ellos.
        """
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            repair_solution(laguna, z_from_plrz, vc)
        bin_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(bin_warns) > 0, (
            "Se esperaban UserWarning por candidatos con demasiados bins, "
            "pero no se emitió ninguno"
        )

    def test_physical_limit_exactly_7_warnings_minimum_set_cover(
        self, laguna, z_from_plrz, vc
    ):
        """Regresión: exactamente 7 candidatos violan max_bins=8 con z del set cover."""
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            repair_solution(laguna, z_from_plrz, vc)
        n_warns = sum(1 for w in caught if issubclass(w.category, UserWarning))
        assert n_warns == 7, (
            f"Se esperaban 7 warnings de exceso de bins, obtenidos {n_warns}"
        )

    def test_physical_limit_known_violating_candidates(
        self, laguna, z_from_plrz, vc
    ):
        """Regresión: los candidatos j=6,10,17,19,24,28,56 superan max_bins con z escaso."""
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            fs = repair_solution(laguna, z_from_plrz, vc)
        known = {6, 10, 17, 19, 24, 28, 56}
        for j in known:
            total = int(np.sum(fs.x[j, :]))
            assert total > laguna.params.max_bins, (
                f"j={j} debería superar max_bins={laguna.params.max_bins}, "
                f"obtenido total={total}"
            )

    def test_physical_limit_satisfied_all_open(self, laguna, repair_all_open):
        """Con todos los candidatos abiertos, ningún j supera max_bins=8."""
        x = repair_all_open.x
        max_bins = laguna.params.max_bins
        totals = np.sum(x, axis=1)
        violations = np.where(totals > max_bins)[0]
        assert len(violations) == 0, (
            f"{len(violations)} candidatos superan max_bins={max_bins} con z=all-True: "
            f"{violations.tolist()}"
        )

    def test_physical_limit_all_open_max_is_bounded(self, laguna, repair_all_open):
        """Con z=all-True, el máximo de bins por candidato es exactamente max_bins."""
        x = repair_all_open.x
        max_total = int(np.max(np.sum(x, axis=1)))
        assert max_total <= laguna.params.max_bins, (
            f"max total bins per candidate={max_total} > max_bins={laguna.params.max_bins}"
        )

    # -----------------------------------------------------------------------
    # 7. w consistente con x: w[j,k] == (x[j,k] > 0)
    # -----------------------------------------------------------------------

    def test_w_consistent_with_x_exactly(self, repair_zero):
        """w == (x > 0) elemento a elemento — identidad exacta."""
        w = repair_zero.w
        x = repair_zero.x
        expected = (x > 0)
        n_diff = int(np.sum(w != expected))
        assert n_diff == 0, (
            f"w y (x > 0) difieren en {n_diff} posiciones"
        )

    def test_w_false_implies_x_zero(self, repair_zero):
        """w[j,k]=False → x[j,k]=0 sin excepciones."""
        w = repair_zero.w
        x = repair_zero.x
        n_viol = int(np.sum(x[~w] > 0))
        assert n_viol == 0, (
            f"x tiene {n_viol} valores >0 donde w=False"
        )

    def test_w_true_implies_x_positive(self, repair_zero):
        """w[j,k]=True → x[j,k]>0 sin excepciones."""
        w = repair_zero.w
        x = repair_zero.x
        n_viol = int(np.sum(x[w] == 0))
        assert n_viol == 0, (
            f"x tiene {n_viol} ceros donde w=True"
        )

    def test_w_consistent_all_open(self, repair_all_open):
        """w == (x > 0) también con z=all-True."""
        assert np.array_equal(repair_all_open.w, (repair_all_open.x > 0)), (
            "w y (x > 0) difieren con z=all-True"
        )

    # -----------------------------------------------------------------------
    # 8. Coste correcto: cost = Σ opening_cost·z + Σ bin_cost[k]·x[j,k]
    # -----------------------------------------------------------------------

    def test_cost_equals_fixed_plus_variable(self, laguna, repair_zero):
        """cost = Σ_j opening_cost·z[j] + Σ_{j,k} bin_cost[k]·x[j,k]."""
        z = repair_zero.z
        x = repair_zero.x
        fixed = float(np.sum(z) * laguna.params.opening_cost)
        bin_cost = np.array([laguna.params.bin_cost[k] for k in laguna.K])
        variable = float(np.sum(x * bin_cost))
        expected = fixed + variable
        assert repair_zero.cost == pytest.approx(expected, rel=1e-9), (
            f"cost={repair_zero.cost:.4f}, esperado={expected:.4f} "
            f"(fixed={fixed:.2f}, variable={variable:.2f})"
        )

    def test_cost_fixed_part_proportional_to_n_open(self, laguna, repair_zero):
        """La parte fija del coste == opening_cost × n_open."""
        z = repair_zero.z
        n_open = int(np.sum(z))
        bin_cost = np.array([laguna.params.bin_cost[k] for k in laguna.K])
        variable = float(np.sum(repair_zero.x * bin_cost))
        expected = n_open * laguna.params.opening_cost + variable
        assert repair_zero.cost == pytest.approx(expected, rel=1e-9)

    def test_cost_positive(self, repair_zero):
        """El coste de cualquier solución con al menos un candidato abierto > 0."""
        assert repair_zero.cost > 0.0, f"cost={repair_zero.cost} debe ser > 0"

    def test_regression_cost_minimum_set_cover(self, repair_zero):
        """Regresión: cost = 168 000 € con z del minimum set cover (27 candidatos)."""
        assert repair_zero.cost == pytest.approx(168_000.0, rel=1e-9), (
            f"cost={repair_zero.cost}, esperado=168 000.0"
        )

    def test_regression_cost_all_open(self, repair_all_open):
        """Regresión: cost = 683 900 € con todos los 133 candidatos abiertos."""
        assert repair_all_open.cost == pytest.approx(683_900.0, rel=1e-9), (
            f"cost={repair_all_open.cost}, esperado=683 900.0"
        )

    # -----------------------------------------------------------------------
    # 9. x[j,k] == 0 para candidatos cerrados
    # -----------------------------------------------------------------------

    def test_no_bins_at_closed_candidates(self, repair_zero):
        """x[j,k] == 0 para todo j con z[j]=False."""
        z = repair_zero.z
        x = repair_zero.x
        n_viol = int(np.sum(x[~z] > 0))
        assert n_viol == 0, (
            f"{n_viol} bins asignados a candidatos cerrados"
        )

    def test_no_bins_at_closed_per_waste_type(self, laguna, repair_zero):
        """Para cada k, x[j,k] == 0 en todos los candidatos cerrados."""
        z = repair_zero.z
        x = repair_zero.x
        for k in range(laguna.n_waste_types):
            n_viol = int(np.sum(x[~z, k] > 0))
            assert n_viol == 0, (
                f"k={k}: {n_viol} candidatos cerrados con x[j,{k}] > 0"
            )

    def test_w_false_at_all_closed_candidates(self, repair_zero):
        """w[j,k]=False para todo j con z[j]=False — coherente con x=0."""
        z = repair_zero.z
        w = repair_zero.w
        n_viol = int(np.sum(w[~z]))
        assert n_viol == 0, (
            f"{n_viol} posiciones (j,k) con j cerrado tienen w=True"
        )

    def test_no_bins_at_closed_candidates_all_open_trivially(self, repair_all_open):
        """Con z=all-True no hay candidatos cerrados — x puede ser no nulo en todos."""
        z = repair_all_open.z
        assert np.all(z), "Precondición: z=all-True debe tener todos True"
        # Verificación de coherencia: x >= 0 en todos los puntos
        assert np.all(repair_all_open.x >= 0)

    # -----------------------------------------------------------------------
    # 10. Con z=all-True, y_assign coincide con valid_candidates[i][k][0]
    # -----------------------------------------------------------------------

    def test_all_open_assigns_nearest_absolute(self, laguna, vc, repair_all_open):
        """
        Con todos los candidatos abiertos, el más cercano abierto es vc[i][k][0]
        (el más cercano absoluto). y_assign[i,k] debe igualar vc[i][k][0].
        """
        ya = repair_all_open.y_assign
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                if vc[i][k]:
                    assert int(ya[i, k]) == vc[i][k][0], (
                        f"(i={i},k={k}): y_assign={ya[i,k]}, "
                        f"vc[i][k][0]={vc[i][k][0]}"
                    )

    def test_all_open_distance_is_global_minimum_in_dij(
        self, laguna, vc, repair_all_open
    ):
        """
        Con z=all-True, la distancia al j asignado es la mínima en vc[i][k]
        (orden ASC garantizado por precompute_valid_candidates).
        """
        ya = repair_all_open.y_assign
        for i in range(laguna.n_buildings):
            for k in range(laguna.n_waste_types):
                j = int(ya[i, k])
                if j == -1:
                    continue
                d_assigned = laguna.dij[j][i]
                for j2 in vc[i][k]:
                    d2 = laguna.dij[j2][i]
                    assert d_assigned <= d2 + 1e-9, (
                        f"(i={i},k={k}): d={d_assigned:.4f} > d(j2={j2})={d2:.4f} "
                        f"— no es el mínimo global en dij"
                    )

    def test_all_open_full_coverage(self, laguna, repair_all_open):
        """Con z=all-True sobre instancia Laguna, cobertura = 100% (cero -1)."""
        n_minus1 = int(np.sum(repair_all_open.y_assign == -1))
        assert n_minus1 == 0, (
            f"{n_minus1} pares (i,k) sin asignación con z=all-True"
        )

    def test_all_open_y_assign_in_range(self, laguna, repair_all_open):
        """Con z=all-True, y_assign tiene solo índices en [0, n_candidates)."""
        ya = repair_all_open.y_assign
        n_j = laguna.n_candidates
        assert np.all(ya >= 0), "z=all-True: y_assign tiene -1 inesperado"
        assert np.all(ya < n_j), (
            f"z=all-True: y_assign tiene índices ≥ n_candidates={n_j}"
        )

    def test_all_open_no_mismatch_with_vc_first(self, laguna, vc, repair_all_open):
        """Regresión: 0 discrepancias entre y_assign y vc[i][k][0] con z=all-True."""
        ya = repair_all_open.y_assign
        mismatches = sum(
            1
            for i in range(laguna.n_buildings)
            for k in range(laguna.n_waste_types)
            if vc[i][k] and int(ya[i, k]) != vc[i][k][0]
        )
        assert mismatches == 0, (
            f"{mismatches} discrepancias entre y_assign y vc[i][k][0] con z=all-True"
        )
