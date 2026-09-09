"""QSVT response families, realisations and refit (2026-09-07 additions).

Contracts:
- ``filter_design`` selector == the pipeline's hand-built polynomial path;
  the default is "comb" (the QSVT of the papers, George 2026-09-07); ``s``
  is read off a constant diagonal.
- The numpy-only quantum lane imports with qiskit blocked; the circuit
  classes fail lazily with a clear message.
- normalized comb / one-bit cosine == the originating study code verbatim.
- phase sequence: scalar identity + dense circuit reproduce q(X) u; the
  combined even/odd solve reproduces the LCU amplitudes.
- refit: moments == Chebyshev recursion; the refit polynomial re-applied
  through QSVT reproduces its own amplitudes; the selection score equals the
  study's ``eff_f010`` column of ``frontier_metrics``.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.polynomial import chebyshev as npcheb

from lhcb_velo_toy.analysis import truth_from_event
from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers.hamiltonians.fast import SimpleHamiltonianFast
from lhcb_velo_toy.solvers.quantum import (
    DEFAULT_DESIGN,
    QSVT,
    apply_phase_sequence,
    apply_response,
    chebyshev_moments,
    design_line_comb_inverse,
    design_normalized_comb,
    design_one_bit_cosine,
    design_trig_comb,
    find_phase_factors,
    mode_system,
    normalized_coupling,
    one_bit_response,
    p4_lines,
    parity_split,
    qsp_signal,
    refit_binned,
    refit_coefficients,
    solve_normalized_comb,
    spectral_bounds,
)
from lhcb_velo_toy.analysis.segment_metrics import compute_epsilon

N_MODULES, DZ, Z0, HALF = 5, 33.0, 100.0, 50.0


def build(T, me, cn, seed, gamma=3.0, delta=1.0):
    geo = PlaneGeometry(
        module_id=list(range(N_MODULES)), lx=[HALF] * N_MODULES,
        ly=[HALF] * N_MODULES, z=[Z0 + i * DZ for i in range(N_MODULES)],
    )
    gen = StateEventGenerator(
        detector_geometry=geo, events=1, n_particles=[T],
        measurement_error=me, collision_noise=cn, seed=seed,
    )
    gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 50.0})
    gen.generate_particles([[{"type": "pion", "mass": 139.6, "q": 1}] * T])
    gen.generate_complete_events()
    ham = SimpleHamiltonianFast(
        epsilon=compute_epsilon(me, cn, dz=DZ), gamma=gamma, delta=delta
    )
    A, b = ham.construct_hamiltonian(gen)
    return gen, ham, A, b


# ---------------------------------------------------------------------------
# constructor: design selector, default, s inference
# ---------------------------------------------------------------------------
def test_default_design_is_the_comb():
    _, _, A, b = build(5, 0.0, 1e-8, 101)
    assert DEFAULT_DESIGN == "comb"
    q0 = QSVT(A, b, degree=40)
    q1 = QSVT(A, b, degree=40, filter_design="comb")
    assert q0.filter_design == "comb" and q0.s == 4.0
    assert np.allclose(q0.coeffs, q1.coeffs)
    assert np.allclose(q0.coeffs, QSVT.comb(A, b).coeffs)
    # the retired band design is still selectable and distinct
    qb = QSVT(A, b, degree=40, filter_design="band")
    assert qb.filter_design == "band" and not np.allclose(qb.coeffs, q0.coeffs)
    _, p = qb.solve_statevector()
    assert p == pytest.approx(0.03702804240383503, rel=1e-6)


def test_comb_selector_equals_pipeline_path():
    _, _, A, b = build(20, 0.01, 1e-4, 202)
    q = QSVT(A, b, degree=40, filter_design="comb")
    assert q.filter_design == "comb"
    p = design_line_comb_inverse(degree=40, s=4.0, domain=tuple(q.poly.domain))
    q2 = QSVT(A, b, poly=p)
    assert np.allclose(q.coeffs, q2.coeffs)
    s1, p1 = q.solve_statevector()
    s2, p2 = q2.solve_statevector()
    assert p1 == pytest.approx(p2, rel=1e-12)
    assert np.allclose(s1, s2)
    # classmethod shorthand
    q3 = QSVT.comb(A, b)
    assert q3.degree == 40 and np.allclose(q3.coeffs, q.coeffs)


def test_s_is_read_off_the_diagonal():
    _, _, A, b = build(5, 0.0, 1e-8, 101, gamma=1.0, delta=1.0)
    q = QSVT(A, b, degree=40, filter_design="comb")
    assert q.s == 2.0
    lines = p4_lines(2.0)
    p_expect = design_line_comb_inverse(degree=40, s=2.0, lines=lines,
                                        domain=tuple(q.poly.domain))
    assert np.allclose(q.coeffs, p_expect.coef)
    assert QSVT(A, b, degree=40, s=4.0).s == 4.0   # explicit override wins


def test_bad_selector_and_double_spec():
    _, _, A, b = build(5, 0.0, 1e-8, 101)
    with pytest.raises(ValueError):
        QSVT(A, b, filter_design="nope")
    with pytest.raises(ValueError):
        QSVT(A, b, poly=design_line_comb_inverse(), filter_design="comb")


def test_all_designs_realisable_and_solve():
    _, _, A, b = build(20, 0.01, 1e-4, 202)
    for design in ("band", "comb", "minimax", "cosine"):
        q = QSVT(A, b, degree=40 if design != "minimax" else 8, filter_design=design)
        x = np.linspace(-1, 1, 4001)
        assert np.max(np.abs(npcheb.chebval(x, q.coeffs))) <= 1.0 + 1e-9
        sol, p = q.solve_statevector()
        assert sol.shape == (A.shape[0],) and 0.0 < p <= 1.0
        sol_e, p_e = q.solve_eigh()
        assert np.allclose(sol, sol_e, atol=1e-9) and p_e == pytest.approx(p, rel=1e-9)


def test_resources_three_realisations():
    _, _, A, b = build(5, 0.0, 1e-8, 101)
    q = QSVT(A, b, degree=40, filter_design="comb")
    r = q.resource_table()
    assert r["lcu"]["ancillas"] == 7 and r["lcu"]["calls_per_shot"] == 63
    assert r["lcu"]["calls_per_shot_optimal_select"] == 40
    assert r["phase"]["ancillas"] == 4 and r["phase"]["calls_per_shot"] == 160
    assert r["phase"]["parity"] == "mixed"
    assert r["qetu"]["ancillas"] == 2 and r["qetu"]["calls_per_shot"] == 40
    assert r["lcu"]["lcu_penalty"] == pytest.approx((q.l1 / r["lcu"]["peak"]) ** 2)
    # legacy keys survive
    assert q.resources()["total_qubits"] == q.total_qubits
    assert q.resources()["walk_calls"] == 40


# ---------------------------------------------------------------------------
# import without qiskit
# ---------------------------------------------------------------------------
def test_quantum_lane_imports_without_qiskit():
    code = textwrap.dedent("""
        import sys, importlib.abc
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "qiskit" or name.startswith("qiskit"):
                    raise ImportError("qiskit blocked for the test")
                return None
        sys.meta_path.insert(0, Block())
        for m in [k for k in sys.modules if k.startswith("qiskit")]:
            del sys.modules[m]
        from lhcb_velo_toy.solvers.quantum import QSVT, solve_one_bit_matrixfree
        from lhcb_velo_toy import QSVT as Q2
        assert Q2 is QSVT
        import lhcb_velo_toy.solvers.quantum as Q
        try:
            Q.OneBQF
        except ImportError as e:
            assert "quantum" in str(e)
        else:
            raise SystemExit("OneBQF should not import without qiskit")
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


# ---------------------------------------------------------------------------
# response families == originating study code
# ---------------------------------------------------------------------------
def _study_design_norm_comb(degree, hw=0.10):
    x = np.linspace(-1, 1, 8000)
    y = np.zeros_like(x)
    for m in (0.5, -0.5):
        y = np.maximum(y, np.exp(-(((x - m) / hw) ** 2)))
    p = npcheb.Chebyshev.fit(x, y, degree, domain=[-1, 1])
    mx = float(np.max(np.abs(p(x))))
    return p / (mx / 0.95) if mx > 1.0 else p


def _study_discriminant(C):
    C = sp.csr_matrix(abs(C))
    deg = np.asarray(C.sum(1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-300)), 0.0)
    return sp.csr_matrix(sp.diags(inv) @ C @ sp.diags(inv))


def _study_cheb_apply(poly, A, b):
    lo, hi = poly.domain
    sc, sh = 2.0 / (hi - lo), (lo + hi) / (hi - lo)
    c = poly.coef
    t_prev = b
    y = c[0] * t_prev
    if len(c) > 1:
        t_cur = sc * (A @ b) - sh * b
        y = y + c[1] * t_cur
        for k in range(2, len(c)):
            t_next = 2.0 * (sc * (A @ t_cur) - sh * t_cur) - t_prev
            t_prev, t_cur = t_cur, t_next
            y = y + c[k] * t_cur
    return np.abs(y)


def test_normalized_comb_matches_study():
    _, _, A, b = build(20, 0.01, 1e-4, 202)
    Off = A.copy().tolil()
    Off.setdiag(0)
    Dm = _study_discriminant(Off.tocsr())
    assert np.allclose(normalized_coupling(A).toarray(), Dm.toarray())
    w = np.linalg.eigvalsh(Dm.toarray())
    assert w[0] >= -1 - 1e-9 and w[-1] <= 1 + 1e-9
    for d in (12, 40):
        ref = _study_cheb_apply(_study_design_norm_comb(d), Dm, np.ones(Dm.shape[0]))
        ours, p = solve_normalized_comb(A, degree=d)
        assert np.allclose(ours, ref)
        assert np.allclose(p.coef, _study_design_norm_comb(d).coef)
        assert np.allclose(p.coef, design_normalized_comb(d).coef)


def test_one_bit_cosine_matches_study_and_exact_response():
    _, _, A, b = build(20, 0.01, 1e-4, 202)
    lo, hi = spectral_bounds(A, pad=0.0)
    pad = 0.05 * (hi - lo) + 0.2
    dom = (lo - pad, hi + pad)
    gr = np.linspace(dom[0], dom[1], 4000)
    u = (2 * gr - (dom[0] + dom[1])) / (dom[1] - dom[0])
    cf = npcheb.chebfit(u, np.abs(np.cos(np.pi * gr / (2 * 4.0))), 60)
    p_study = npcheb.Chebyshev(cf, domain=list(dom))
    p = design_one_bit_cosine(degree=60, s=4.0, domain=dom)
    assert np.allclose(p.coef, p_study.coef)
    ref = _study_cheb_apply(p_study, A, b)
    q = QSVT(A, b, poly=p)
    sol, _ = q.solve_statevector()
    assert np.allclose(sol, ref / np.linalg.norm(ref))
    # against the exact cosine through the mode system
    sysd = mode_system(A, b)
    exact = apply_response(sysd, one_bit_response(s=4.0))
    cos = float(sol @ exact / (np.linalg.norm(sol) * np.linalg.norm(exact)))
    # |cos| has a kink at the notch: the degree-60 fit of the study is ~0.998
    assert cos > 0.99
    # the signed cosine (|cos(At/2) b|, the real-cosine idealisation) is a
    # different object from the |cos| response applied mode by mode; it is
    # smooth and its degree-60 realisation is exact to round-off
    ps = design_one_bit_cosine(degree=60, s=4.0, domain=dom, signed=True)
    sol_s, _ = QSVT(A, b, poly=ps).solve_statevector()
    exact_s = np.abs(apply_response(sysd, one_bit_response(s=4.0, signed=True),
                                    signed=True))
    cos_s = float(sol_s @ exact_s / (np.linalg.norm(sol_s) * np.linalg.norm(exact_s)))
    assert cos_s > 1 - 1e-8
    assert cos_s > cos
    # the notch: the isolated line at s is zeroed
    assert one_bit_response(4.0)(np.array([4.0]))[0] == pytest.approx(0.0, abs=1e-15)


def test_trig_comb_is_a_bounded_comb():
    f = design_trig_comb(K=40)
    assert f.K == 40 and f.coef.shape == (41,)
    x = np.linspace(0.2, 7.8, 6000)
    assert np.max(np.abs(f(x))) == pytest.approx(1.0, rel=1e-9)
    lines = np.array(p4_lines(4.0))
    assert np.min(np.abs(f(lines))) > 0.3          # passes at the true lines
    assert abs(f(np.array([4.0]))[0]) < 0.05         # ~0 at the isolated line
    assert abs(f(np.array([3.0]))[0]) < 0.1          # ~0 at the pair line s-1


# ---------------------------------------------------------------------------
# phase sequence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d,seed", [(8, 1), (9, 2)])
def test_phase_factors_scalar_and_circuit(d, seed):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=d + 1)
    if d % 2 == 0:
        c[1::2] = 0
    else:
        c[0::2] = 0
    xg = np.linspace(-1, 1, 2001)
    c /= np.max(np.abs(npcheb.chebval(xg, c)))
    phis, err = find_phase_factors(c)
    assert len(phis) == d + 1 and err < 1e-4
    assert np.max(np.abs(qsp_signal(phis, xg).real - npcheb.chebval(xg, c))) < 1e-4
    n = 10
    M = rng.normal(size=(n, n))
    M = (M + M.T) / 2
    X = M / np.linalg.norm(M, 2) * 0.9
    u = rng.normal(size=n)
    u /= np.linalg.norm(u)
    w, V = np.linalg.eigh(X)
    ref = (V * npcheb.chebval(w, c)) @ V.T @ u
    psi = apply_phase_sequence(X, u, phis)
    assert np.max(np.abs(psi.real - ref)) < 1e-4


def test_parity_split_and_penalty():
    p = design_line_comb_inverse(degree=40, s=4.0, domain=(0.2, 7.8))
    parts = parity_split(p.coef)
    assert parts["scale"] == pytest.approx(0.3494355, rel=1e-4)      # paper: max|p| = 0.349
    assert parts["alpha_e"] == pytest.approx(0.70752, rel=1e-3)
    assert parts["alpha_o"] == pytest.approx(0.29290, rel=1e-3)
    assert parts["parity_penalty"] == pytest.approx(1.001, abs=2e-3)  # paper: 1.001
    x = np.linspace(-1, 1, 5001)
    assert np.max(np.abs(npcheb.chebval(x, parts["ce"]))) == pytest.approx(1.0, rel=1e-4)
    assert np.max(np.abs(npcheb.chebval(x, parts["co"]))) == pytest.approx(1.0, rel=1e-4)
    with pytest.raises(ValueError):
        find_phase_factors(parts["c"])   # mixed parity


def test_solve_phase_sequence_matches_lcu_amplitudes():
    _, _, A, b = build(5, 0.0, 1e-8, 101)          # 100 segments, dense OK
    q = QSVT(A, b, degree=12, filter_design="comb")
    sol_lcu, p_lcu = q.solve_statevector()
    sol_ph, p_ph = q.solve_phase_sequence()
    assert np.max(np.abs(sol_ph - sol_lcu)) < 1e-3
    assert max(q.phase_info["max_err"].values()) < 1e-4
    # success probability of the phase sequence: ||q(X)b||^2/(ae+ao)^2 with
    # q = p/max|p|, i.e. the LCU probability times (l1/peak)^2 / penalty
    r = q.resources("phase")
    expect = p_lcu * r["lcu_penalty"] / q.phase_info["parity_penalty"]
    assert p_ph == pytest.approx(expect, rel=1e-3)
    assert p_ph > p_lcu
    # reuse of phases
    sol2, p2 = q.solve_phase_sequence(phases=q.phase_info["phases"])
    assert np.allclose(sol2, sol_ph) and p2 == pytest.approx(p_ph)
    with pytest.raises(ValueError):
        q.solve_phase_sequence(max_dim=10)


# ---------------------------------------------------------------------------
# refit + matched-efficiency metrics
# ---------------------------------------------------------------------------
def _study_frontier_metrics(x, truth, x_cls, EFF_TARGETS=(0.99, 0.995, 0.999),
                            FAR_TARGETS=(0.001, 0.01)):
    x = np.abs(np.asarray(x).ravel())
    truth = np.asarray(truth, bool)
    n_true = int(truth.sum())
    ts = np.sort(x[truth])[::-1]
    out = {}
    for e in EFF_TARGETS:
        k = min(n_true, max(1, int(np.ceil(e * n_true))))
        tau = ts[k - 1]
        act = x >= tau
        na = int(act.sum())
        tag = f"e{int(round(e * 1000)):03d}"
        out[f"far_{tag}"] = int((act & ~truth).sum()) / na if na else np.nan
        out[f"eff_{tag}"] = float((act & truth).sum()) / n_true
    taus = ts[np.arange(1, n_true + 1) - 1]
    xs, xt = np.sort(x), np.sort(x[truth])
    nact = len(x) - np.searchsorted(xs, taus, side="left")
    ntru = n_true - np.searchsorted(xt, taus, side="left")
    far = np.where(nact > 0, (nact - ntru) / np.maximum(nact, 1), np.nan)
    eff = ntru / n_true
    for ft in FAR_TARGETS:
        okm = far <= ft
        out[f"eff_f{int(round(ft * 1000)):03d}"] = (
            float(eff[okm].max()) if okm.any() else 0.0)
    nx, nc = np.linalg.norm(x), np.linalg.norm(x_cls)
    out["cos_C"] = float(x @ np.abs(x_cls) / (nx * nc)) if nx * nc > 0 else np.nan
    return out


def test_refit_selection_score_equals_study_column():
    from lhcb_velo_toy.solvers.quantum.refit import _eff_at_far

    gen, ham, A, b = build(20, 0.01, 1e-4, 202)
    truth = truth_from_event(gen)
    rng = np.random.default_rng(3)
    x_cls = np.abs(ham.solve_classicaly())
    x = x_cls + 0.05 * rng.standard_normal(len(x_cls))
    theirs = _study_frontier_metrics(x, truth, x_cls)
    assert _eff_at_far(x, truth, 0.01) == pytest.approx(theirs["eff_f010"], abs=1e-12)
    assert _eff_at_far(x, truth, 0.001) == pytest.approx(theirs["eff_f001"], abs=1e-12)


def test_chebyshev_moments_equal_recursion_and_refit_roundtrip():
    gen, ham, A, b = build(20, 0.01, 1e-4, 202)
    truth = truth_from_event(gen)
    x_cls = np.abs(ham.solve_classicaly())
    lo, hi = spectral_bounds(A, pad=0.0)
    c0, c1 = 0.5 * (hi + lo), 0.5 * (hi - lo) * 1.02
    dom = (c0 - c1, c0 + c1)
    M = chebyshev_moments(A, b, dom, dmax=24, dtype=np.float64)
    # column k must equal T_k(X) b as the QSVT recursion defines it
    q = QSVT(A, b, poly=npcheb.Chebyshev(np.eye(25)[24], domain=list(dom)),
             spectrum_pad=0.0)
    q.solve_statevector()
    y24 = q.solution_signed * np.linalg.norm(b)   # T_24(X) b / ||b|| -> rescale
    assert np.allclose(np.abs(M[:, 24]) / np.linalg.norm(M[:, 24]),
                       np.abs(y24) / np.linalg.norm(y24))
    res = refit_coefficients(M, truth, x_cls, degree=16, domain=dom)
    assert res.degree == 16 and res.coef.shape == (17,)
    assert res.mu > 0 and res.far_target == 0.01
    assert 0.0 <= res.eff_at_far <= 1.0
    # transfer path: the refit polynomial through QSVT reproduces its amplitudes
    q2 = QSVT(A, b, poly=res.poly, spectrum_pad=0.0)
    sol, _ = q2.solve_statevector()
    amp = res.amplitudes / np.linalg.norm(res.amplitudes)
    assert np.allclose(sol, amp, atol=1e-6)
    assert q2.filter_design == "custom"


def test_refit_binned_runs_on_mode_system():
    gen, ham, A, b = build(20, 0.01, 1e-4, 202)
    truth = truth_from_event(gen)
    x_cls = np.abs(ham.solve_classicaly())
    sysd = mode_system(A, b, iso_lambda=4.0)
    p_bins, edges, mu = refit_binned(sysd, truth, x_cls)
    assert len(edges) == len(p_bins) + 1
    assert np.max(np.abs(p_bins)) == pytest.approx(1.0)
    assert mu > 0
