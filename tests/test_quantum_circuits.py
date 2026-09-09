"""The three qiskit reference circuits under the current qiskit (2.x) API.

Contracts:
- No DeprecationWarning attributed to ``lhcb_velo_toy`` while building and
  running HHL, OneBQF (sampling + statevector) and the QSVT circuit (the
  pytest config also turns such warnings into errors suite-wide).
- 1BQF uncompute (2026-09-07 fix): the inverse controlled evolution is the
  exact inverse of the forward one (forward-order application is not); the
  circuit readout matches the exact-evolution readout of the paper,
  p_i = [cos^2(At/2) u]_i^2 + [sin(At) u / 2]_i^2, to a cosine > 0.999 (the
  residual is the per-pair product formula for e^{-iAt}, which is order
  dependent at the ~1 % level — a physics caveat, not the uncompute).
- OneBQF sampling readout == statevector readout up to shot noise; dense and
  sparse inputs give the same circuit.
- HHL builds, runs, and its two readout paths return unit-norm vectors of
  the right dimension (its known gain / keying / readout defects are a
  separate Notion to-do and are NOT asserted here).
- QSVT explicit qiskit circuit == matrix-free backend.
"""
import subprocess
import sys
import textwrap
from pathlib import Path
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

qiskit = pytest.importorskip("qiskit")
from qiskit import transpile  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402

from lhcb_velo_toy.solvers.quantum import (  # noqa: E402
    QSVT,
    HHLAlgorithm,
    OneBitHHL,
    OneBQF,
    solve_one_bit_matrixfree,
)
from lhcb_velo_toy.solvers.quantum.matrixfree import _controlled_U  # noqa: E402
from tests.test_golden_baseline import build  # noqa: E402

CHAIN4 = np.array([[2.0, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]])
B4 = np.array([1.0, 0, 0, 1])


def _statevector(qc):
    sim = AerSimulator(method="statevector")
    return sim.run(transpile(qc, sim)).result().get_statevector()


def _our_warnings(records):
    return [r for r in records if "lhcb_velo_toy" in str(r.filename)
            and issubclass(r.category, (DeprecationWarning, PendingDeprecationWarning))]


# ---------------------------------------------------------------------------
# qiskit API currency
# ---------------------------------------------------------------------------
def test_circuits_raise_no_deprecations_from_our_code():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o = OneBQF(CHAIN4, B4, shots=500)
        o.build_circuit()
        o.run()
        o.get_solution()
        o2 = OneBQF(CHAIN4, B4, readout="statevector")
        o2.get_solution_from_statevector(_statevector(o2.build_circuit()))
        h = HHLAlgorithm(CHAIN4, B4, num_time_qubits=3, shots=500)
        h.build_circuit()
        h.run()
        h.get_solution()
        h.extract_postselected_solution(h.simulate_statevector())
        q = QSVT(CHAIN4, B4, degree=8)
        q.build_circuit()
        q.run_circuit()
    ours = _our_warnings(w)
    assert not ours, "\n".join(f"{r.filename}:{r.lineno}: {r.message}" for r in ours)


def test_package_class_survives_submodule_import_order():
    """`from ...quantum import OneBQF` is the CLASS whatever was imported first.

    Regression (2026-09-07): the module was named ``OneBQF.py``, the same name
    as the class the package exports lazily. Importing the submodule — which
    the documented ``one_bit_hhl`` compatibility shim does — made the import
    machinery bind the MODULE over the class, so the documented import handed
    back something that was not callable. The module is now ``one_bqf``.
    Needs a fresh interpreter: this is about import order.
    """
    code = textwrap.dedent("""
        from lhcb_velo_toy.solvers.quantum.one_bit_hhl import OneBitHHL   # shim FIRST
        from lhcb_velo_toy.solvers.quantum import OneBQF, OneBitHHL as Alias
        assert isinstance(OneBQF, type), f"expected class, got {type(OneBQF)}"
        assert callable(OneBQF) and OneBQF is Alias is OneBitHHL
        import lhcb_velo_toy.solvers.quantum.one_bqf                      # direct submodule
        import lhcb_velo_toy.solvers.quantum as Q
        assert isinstance(Q.OneBQF, type), f"clobbered: {type(Q.OneBQF)}"
        import lhcb_velo_toy.solvers.quantum.QSVT                         # eager pair
        assert isinstance(Q.QSVT, type), f"clobbered: {type(Q.QSVT)}"
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


def test_no_module_names_differ_only_by_case():
    """Two modules whose names differ only in case collapse into one file on a
    case-insensitive filesystem (macOS, Windows), so a clone there silently
    loses one of them.

    Regression (2026-09-09): ``HHL.py`` (the implementation) sat beside
    ``hhl.py`` (a back-compat shim re-exporting it). They are now one module,
    ``hhl.py``. This is the same class of defect as the ``OneBQF.py`` /
    ``OneBQF`` collision above, one level down: the filesystem rather than the
    import machinery.
    """
    root = Path(__file__).resolve().parent.parent / "src" / "lhcb_velo_toy"
    clashes = []
    for d in {p.parent for p in root.rglob("*.py")}:
        seen = {}
        for f in d.glob("*.py"):
            seen.setdefault(f.name.lower(), []).append(f.name)
        clashes += [(str(d.relative_to(root)), names)
                    for names in seen.values() if len(names) > 1]
    assert not clashes, f"module names differing only by case: {clashes}"


def test_hhl_import_paths():
    """The documented HHL import works, and the retired shim path is gone."""
    from lhcb_velo_toy.solvers.quantum import HHLAlgorithm
    from lhcb_velo_toy.solvers.quantum.hhl import HHLAlgorithm as Direct
    assert HHLAlgorithm is Direct and isinstance(HHLAlgorithm, type)


def test_alias_and_lazy_names():
    assert OneBitHHL is OneBQF
    import lhcb_velo_toy.solvers.quantum as Q
    assert "HHLAlgorithm" in dir(Q) and "OneBQF" in dir(Q)


# ---------------------------------------------------------------------------
# 1BQF: uncompute fix
# ---------------------------------------------------------------------------
def _pairs_and_params(A):
    A = A.tocsr()
    d = A.shape[0]
    n_sys = max(1, int(np.ceil(np.log2(d))))
    diag = float(A.diagonal()[0])
    t = np.pi / diag
    off = sp.triu(A + A.T, k=1).tocoo()
    return list(zip(off.row.tolist(), off.col.tolist())), n_sys, diag, t


@pytest.mark.parametrize("regime", ["clean", "moderate"])
def test_inverse_evolution_is_exact_inverse(regime):
    _, _, A, _ = build(regime)
    pairs, n_sys, diag, t = _pairs_and_params(A)
    n = n_sys + 2
    rng = np.random.default_rng(1)
    psi0 = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    psi0 /= np.linalg.norm(psi0)
    # reversed order (the fix): identity to round-off
    psi = psi0.copy()
    _controlled_U(psi, pairs, 2 * t, -diag * t, n_sys)
    _controlled_U(psi, pairs[::-1], -2 * t, diag * t, n_sys)
    assert np.max(np.abs(psi - psi0)) < 1e-12
    # forward order (the old behaviour): NOT the inverse on multi-pair clusters
    psi = psi0.copy()
    _controlled_U(psi, pairs, 2 * t, -diag * t, n_sys)
    _controlled_U(psi, pairs, -2 * t, diag * t, n_sys)
    assert np.max(np.abs(psi - psi0)) > 1e-3


def _exact_one_bit_readout(A):
    """Paper readout with the clock traced out, on the zero-padded register."""
    A = A.toarray()
    d = A.shape[0]
    n_sys = max(1, int(np.ceil(np.log2(d))))
    N = 1 << n_sys
    Ap = np.eye(N) * A[0, 0]
    Ap[:d, :d] = A
    t = np.pi / A[0, 0]
    w, V = np.linalg.eigh(Ap)
    u = np.ones(N) / np.sqrt(N)
    c2 = (V * np.cos(w * t / 2) ** 2) @ V.T @ u
    s1 = 0.5 * (V * np.sin(w * t)) @ V.T @ u
    p = c2 ** 2 + s1 ** 2
    return np.sqrt(p / p.sum())[:d], float(p.sum())


@pytest.mark.parametrize("regime,min_cos", [("clean", 0.9999), ("moderate", 0.999)])
def test_fixed_circuit_matches_exact_evolution_readout(regime, min_cos):
    _, _, A, _ = build(regime)
    sol, P, _ = solve_one_bit_matrixfree(A)
    ex, P_ex = _exact_one_bit_readout(A)
    assert float(sol.astype(np.float64) @ ex) > min_cos
    # the success probability is a small number and carries the product-formula
    # error in full (~15 % on the clean event); the amplitude profile does not
    assert P == pytest.approx(P_ex, rel=0.25)


def test_circuit_equals_matrixfree_after_fix():
    _, _, A, b = build("moderate")
    ref = OneBQF(A, b, readout="statevector")
    sol_ref, p_ref = ref.get_solution_from_statevector(_statevector(ref.build_circuit()))
    sol_mf, p_mf, _ = solve_one_bit_matrixfree(A)
    assert p_mf == pytest.approx(p_ref, rel=1e-9, abs=1e-12)
    assert float(np.dot(sol_mf.astype(np.float64), sol_ref)) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# OneBQF readouts and input paths
# ---------------------------------------------------------------------------
def test_onebqf_sampling_matches_statevector_readout():
    sv_obj = OneBQF(CHAIN4, B4, readout="statevector")
    sol_sv, p_sv = sv_obj.get_solution_from_statevector(_statevector(sv_obj.build_circuit()))
    samp = OneBQF(CHAIN4, B4, shots=40000)
    samp.build_circuit()
    sim = AerSimulator(seed_simulator=7)
    counts = sim.run(transpile(samp.circuit, sim), shots=40000).result().get_counts()
    sol_s, n_success = samp.get_solution(counts)
    assert sol_s.shape == (4,) and np.linalg.norm(sol_s) == pytest.approx(1.0)
    assert float(sol_s @ sol_sv) > 0.995
    assert n_success / 40000 == pytest.approx(p_sv, abs=0.02)
    assert samp.get_solution(counts)[1] == n_success


def test_onebqf_success_probability_matches_counts():
    """The documented accessor that the example notebook needs (restored 2026-09-07)."""
    o = OneBQF(CHAIN4, B4, shots=8000)
    o.build_circuit()
    counts = o.run()
    sol, n_success = o.get_solution()
    p = o.get_success_probability()
    assert p == pytest.approx(n_success / sum(counts.values()))
    assert 0.0 <= p <= 1.0
    # matches the exact statevector probability up to shot noise
    sv_obj = OneBQF(CHAIN4, B4, readout="statevector")
    _, p_exact = sv_obj.get_solution_from_statevector(_statevector(sv_obj.build_circuit()))
    assert p == pytest.approx(p_exact, abs=0.03)
    with pytest.raises(ValueError):
        OneBQF(CHAIN4, B4).get_success_probability()


def test_onebqf_dense_and_sparse_inputs_agree():
    dense = OneBQF(CHAIN4, B4, readout="statevector")
    sparse = OneBQF(sp.csr_matrix(CHAIN4), B4, readout="statevector")
    assert dense.interaction_pairs == [(0, 1), (1, 2), (2, 3)]
    assert sparse.interaction_pairs == dense.interaction_pairs
    s1, p1 = dense.get_solution_from_statevector(_statevector(dense.build_circuit()))
    s2, p2 = sparse.get_solution_from_statevector(_statevector(sparse.build_circuit()))
    assert np.allclose(s1, s2) and p1 == pytest.approx(p2)


def test_onebqf_rejects_non_constant_diagonal_and_bad_readout():
    with pytest.raises(ValueError):
        OneBQF(np.diag([1.0, 2.0, 3.0, 4.0]), B4)
    with pytest.raises(ValueError):
        OneBQF(CHAIN4, B4, readout="counts")


# ---------------------------------------------------------------------------
# HHL reference circuit (API only; physics defects are a Notion to-do)
# ---------------------------------------------------------------------------
def test_hhl_builds_runs_and_reads_out():
    h = HHLAlgorithm(CHAIN4, B4, num_time_qubits=3, shots=2000)
    qc = h.build_circuit()
    assert qc.num_qubits == 3 + 2 + 1
    counts = h.run()
    assert sum(counts.values()) == 2000
    sol = h.get_solution()
    assert sol.shape == (4,) and np.linalg.norm(sol) == pytest.approx(1.0)
    clean = h.get_quantum_only_circuit()
    assert all(ci.operation.name != "measure" for ci in clean.data)
    sv = h.simulate_statevector()
    sol_sv = h.extract_postselected_solution(sv)
    assert sol_sv.shape == (4,) and np.linalg.norm(sol_sv) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# QSVT explicit circuit
# ---------------------------------------------------------------------------
def test_qsvt_qiskit_circuit_equals_matrixfree():
    _, _, A, b = build("clean")   # 100 segments -> 7 system qubits
    q = QSVT(A, b, degree=12)
    sol_sv, p_sv = q.solve_statevector()
    sol_qk, p_qk = q.run_circuit(streaming=False)
    assert p_qk == pytest.approx(p_sv, rel=1e-9)
    assert np.allclose(sol_qk, sol_sv, atol=1e-9)
    assert q.circuit.num_qubits == q.total_qubits
