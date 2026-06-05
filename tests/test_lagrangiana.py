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
    solve_plr_yw,
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
