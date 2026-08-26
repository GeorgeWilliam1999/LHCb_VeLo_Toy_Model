"""Adapter: LHCb Run-3 VeLo 'Allen'-format event JSONs -> toy-model events.

Provenance
----------
Promoted verbatim from ``Toy_Characterisation/Run3_Verification/
run3_loader.py`` (centralise-2.1 Phase 4, 2026-08-26) — the best-practice
duck-typed adapter from the 2026-08-26 audit, including the left/right
half-detector geometry fix.

The Allen JSON has flat ``x,y,z`` hit arrays, a ``module_prefix_sum`` (hit i
is on module m where prefix_sum[m] <= i < prefix_sum[m+1]) and
``montecarlo.particles`` (columnar; each particle's last column is the list
of its hit indices).

``SimpleHamiltonianFast.construct_segments`` only needs, from an event:
  * ``.modules``  — list of modules **in z order**, each with ``.hit_ids``
  * ``.get_hits_by_ids(ids)`` — hits with ``.x .y .z .hit_id .track_id``
so we duck-type exactly that. Everything downstream (Hamiltonian build,
truth mask, classical/quantum solves, metrics) then works unchanged.
"""
from __future__ import annotations

import json

import numpy as np

__all__ = ["Run3Event", "load_run3_event"]


class _Hit:
    __slots__ = ("x", "y", "z", "hit_id", "track_id", "module_id")

    def __init__(self, hid, x, y, z, mid, tid):
        self.hit_id = int(hid)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.module_id = int(mid)
        self.track_id = int(tid)


class _Module:
    def __init__(self, mid, z, hit_ids):
        self.module_id = int(mid)
        self.z = float(z)
        self.hit_ids = list(hit_ids)


class Run3Event:
    """Minimal duck-typed event for SimpleHamiltonianFast.construct_segments."""

    def __init__(self, modules, hits):
        self.modules = modules                      # z-ordered
        self._hits = {h.hit_id: h for h in hits}

    def get_hits_by_ids(self, hit_ids):
        return [self._hits[i] for i in hit_ids]


def load_run3_event(path):
    """Return (Run3Event, meta dict with n_hits, n_modules, track_id[], module_id[]).

    The VeLo's left/right modules interleave in z (even z-rank = left x<0,
    odd = right x>0); a track stays on ONE side, so its hits step by 2 in
    z-rank. Segments are built WITHIN each half by ordering left and right
    modules separately, separated by an empty module (construct_segments
    skips empty modules, so no spurious cross-half segment is built).
    """
    with open(path) as f:
        d = json.load(f)
    x, y, z = d["x"], d["y"], d["z"]
    ps = d["module_prefix_sum"]
    n_mod = len(ps) - 1
    nh = len(x)

    module_id = np.zeros(nh, dtype=np.int64)
    for m in range(n_mod):
        module_id[ps[m]:ps[m + 1]] = m

    track_id = -np.ones(nh, dtype=np.int64)         # -1 = noise / unassigned
    desc = d["montecarlo"]["description"]
    hcol, kcol = desc.index("hits"), desc.index("key")
    for p in d["montecarlo"]["particles"]:
        key = int(p[kcol])
        for h in p[hcol]:
            track_id[int(h)] = key

    zc = np.array([np.mean(z[ps[m]:ps[m + 1]]) if ps[m + 1] > ps[m] else np.inf
                   for m in range(n_mod)])
    order = [m for m in np.argsort(zc, kind="stable") if np.isfinite(zc[m])]
    hits = [_Hit(i, x[i], y[i], z[i], module_id[i], track_id[i]) for i in range(nh)]

    def mk(m):
        return _Module(m, zc[m], list(range(ps[m], ps[m + 1])))

    left = [mk(m) for m in order[0::2]]
    right = [mk(m) for m in order[1::2]]
    sep = _Module(-1, np.inf, [])
    modules = left + [sep] + right
    return Run3Event(modules, hits), dict(n_hits=nh, n_modules=n_mod,
                                          track_id=track_id, module_id=module_id,
                                          z_centres=zc, n_left=len(left),
                                          n_right=len(right))
