"""
SimpleHamiltonianFast: Optimized Hamiltonian implementation.

This is a performance-optimized implementation using vectorized operations
and COO sparse matrix construction.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import cg, spsolve

from lhcb_velo_toy.solvers.hamiltonians.base import Hamiltonian

if TYPE_CHECKING:
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment
    from lhcb_velo_toy.generation.entities.event import Event
    from lhcb_velo_toy.generation.generators.state_event import StateEventGenerator


class SimpleHamiltonianFast(Hamiltonian):
    """
    Performance-optimized track-finding Hamiltonian.
    
    This implementation uses vectorized NumPy operations and COO sparse
    matrix construction for improved performance on large events.
    
    Parameters
    ----------
    epsilon : float
        Angular tolerance for segment compatibility (radians).
    gamma : float
        Self-interaction penalty coefficient.
    delta : float
        Bias term coefficient.
    theta_d : float, default 1e-4
        Width parameter for ERF-smoothed thresholding.
    
    Attributes
    ----------
    epsilon : float
        Angular tolerance parameter.
    gamma : float
        Self-interaction penalty.
    delta : float
        Bias term.
    theta_d : float
        ERF smoothing width.
    A : csc_matrix
        Hamiltonian matrix after construction.
    b : ndarray
        Bias vector after construction.
    segments : list[Segment]
        All constructed segments.
    n_segments : int
        Number of segments.
    _segment_vectors : ndarray
        Pre-computed normalized direction vectors for all segments.
    _segment_to_hit_ids : dict
        Mapping from segment index to hit IDs for fast lookup.
    
    Examples
    --------
    >>> ham = SimpleHamiltonianFast(epsilon=0.01, gamma=1.5, delta=1.0)
    >>> ham.construct_hamiltonian(event, convolution=True)
    >>> solution = ham.solve_classicaly()
    
    Notes
    -----
    Performance improvements over SimpleHamiltonian:
    - Pre-computed normalized segment vectors
    - Vectorized angle calculations using matrix operations
    - COO matrix construction (faster than LIL for batch insertions)
    - Automatic solver selection based on matrix size
    
    Typical speedup: 5-15x for events with >1000 segments.
    """
    
    def __init__(
        self,
        epsilon: float,
        gamma: float,
        delta: float,
        theta_d: float = 1e-4,
    ) -> None:
        """Initialize the optimized Hamiltonian."""
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = delta
        self.theta_d = theta_d
        self.A: Optional[csc_matrix] = None
        self.b: Optional[np.ndarray] = None
        self.segments: list["Segment"] = []
        self.segments_grouped: list[list["Segment"]] = []
        self.n_segments: int = 0
        # Cached data for fast construction
        self._segment_vectors: Optional[np.ndarray] = None
        self._segment_to_hit_ids: list[tuple[int, int]] = []
        self._group_boundaries: list[int] = []
    
    def construct_segments(
        self,
        event: "Event | StateEventGenerator",
        materialize_segments: bool = True,
    ) -> None:
        """
        Construct segments with pre-computed direction vectors (vectorised).
        
        Parameters
        ----------
        event : Event or StateEventGenerator
            The event containing hits and modules.
        materialize_segments : bool, default True
            If True, build the full Python list of :class:`Segment` objects in
            ``self.segments`` and ``self.segments_grouped`` (preserves the
            historical API at O(N) Python object allocation cost).  If False,
            those lists are left empty and downstream code must consume the
            numpy arrays below directly.  All arrays are populated either way.
        
        Notes
        -----
        Also computes and caches as numpy arrays:
        - ``_segment_vectors`` (n_seg, 3) normalised direction vectors
        - ``_segment_to_hit_ids`` (n_seg, 2) [start_hit_id, end_hit_id]
        - ``_segment_track_ids`` (n_seg, 2) [start_track_id, end_track_id]
        - ``_segment_endpoints_xyz`` (n_seg, 2, 3) start/end coordinates
        - ``_group_boundaries`` list of cumulative segment offsets per module
        """
        from lhcb_velo_toy.solvers.reconstruction.segment import Segment

        evt = getattr(event, "true_event", event)

        vec_chunks: list[np.ndarray] = []
        hit_id_chunks: list[np.ndarray] = []
        track_id_chunks: list[np.ndarray] = []
        endpoint_chunks: list[np.ndarray] = []
        per_module_from_hits: list[list] = []
        per_module_to_hits: list[list] = []
        group_boundaries: list[int] = [0]
        segment_id = 0

        for idx in range(len(evt.modules) - 1):
            from_hits = evt.get_hits_by_ids(evt.modules[idx].hit_ids)
            to_hits = evt.get_hits_by_ids(evt.modules[idx + 1].hit_ids)
            n_from = len(from_hits)
            n_to = len(to_hits)
            per_module_from_hits.append(from_hits)
            per_module_to_hits.append(to_hits)

            if n_from == 0 or n_to == 0:
                group_boundaries.append(segment_id)
                continue

            # Pull endpoint coordinates / ids into small numpy arrays once.
            from_xyz = np.array(
                [(h.x, h.y, h.z) for h in from_hits], dtype=np.float64
            )  # (n_from, 3)
            to_xyz = np.array(
                [(h.x, h.y, h.z) for h in to_hits], dtype=np.float64
            )  # (n_to, 3)
            from_hid = np.fromiter(
                (h.hit_id for h in from_hits), dtype=np.int64, count=n_from
            )
            to_hid = np.fromiter(
                (h.hit_id for h in to_hits), dtype=np.int64, count=n_to
            )
            from_tid = np.fromiter(
                (h.track_id for h in from_hits), dtype=np.int64, count=n_from
            )
            to_tid = np.fromiter(
                (h.track_id for h in to_hits), dtype=np.int64, count=n_to
            )

            # Vectorised pairwise delta -> normalised direction vector.
            # Resulting shape (n_from, n_to, 3) -> flatten in row-major
            # (from-major) order to match itertools.product(from, to).
            d = to_xyz[None, :, :] - from_xyz[:, None, :]
            norms = np.sqrt((d * d).sum(axis=2))  # (n_from, n_to)
            safe = norms > 0.0
            inv = np.where(safe, 1.0 / np.where(safe, norms, 1.0), 0.0)
            vecs = d * inv[:, :, None]
            # Replace zero-norm rows by (0, 0, 1) to match the old behaviour.
            zero_mask = ~safe
            if zero_mask.any():
                vecs[zero_mask] = (0.0, 0.0, 1.0)
            vecs = vecs.reshape(-1, 3)

            # Endpoint coords broadcast (start = from, end = to).
            start_xyz = np.broadcast_to(
                from_xyz[:, None, :], (n_from, n_to, 3)
            ).reshape(-1, 3)
            end_xyz = np.broadcast_to(
                to_xyz[None, :, :], (n_from, n_to, 3)
            ).reshape(-1, 3)
            endpoints = np.stack([start_xyz, end_xyz], axis=1)  # (n, 2, 3)

            start_hid = np.broadcast_to(
                from_hid[:, None], (n_from, n_to)
            ).reshape(-1)
            end_hid = np.broadcast_to(
                to_hid[None, :], (n_from, n_to)
            ).reshape(-1)
            hid_pairs = np.stack([start_hid, end_hid], axis=1)  # (n, 2)

            start_tid = np.broadcast_to(
                from_tid[:, None], (n_from, n_to)
            ).reshape(-1)
            end_tid = np.broadcast_to(
                to_tid[None, :], (n_from, n_to)
            ).reshape(-1)
            tid_pairs = np.stack([start_tid, end_tid], axis=1)  # (n, 2)

            vec_chunks.append(vecs)
            hit_id_chunks.append(hid_pairs)
            track_id_chunks.append(tid_pairs)
            endpoint_chunks.append(endpoints)

            segment_id += n_from * n_to
            group_boundaries.append(segment_id)

        if vec_chunks:
            self._segment_vectors = np.concatenate(vec_chunks, axis=0)
            self._segment_to_hit_ids = np.concatenate(hit_id_chunks, axis=0)
            self._segment_track_ids = np.concatenate(track_id_chunks, axis=0)
            self._segment_endpoints_xyz = np.concatenate(endpoint_chunks, axis=0)
        else:
            self._segment_vectors = np.zeros((0, 3), dtype=np.float64)
            self._segment_to_hit_ids = np.zeros((0, 2), dtype=np.int64)
            self._segment_track_ids = np.zeros((0, 2), dtype=np.int64)
            self._segment_endpoints_xyz = np.zeros((0, 2, 3), dtype=np.float64)

        self.n_segments = segment_id
        self._group_boundaries = group_boundaries

        if materialize_segments:
            # Backward-compatible eager materialisation of Segment objects.
            # This is the slow part for very large events; opt out with
            # materialize_segments=False if you only need the numpy arrays.
            from itertools import product

            segments: list[Segment] = []
            segments_grouped: list[list[Segment]] = []
            sid = 0
            for from_hits, to_hits in zip(
                per_module_from_hits, per_module_to_hits
            ):
                group: list[Segment] = []
                for from_hit, to_hit in product(from_hits, to_hits):
                    seg = Segment(
                        hit_start=from_hit,
                        hit_end=to_hit,
                        segment_id=sid,
                    )
                    group.append(seg)
                    segments.append(seg)
                    sid += 1
                segments_grouped.append(group)
            self.segments = segments
            self.segments_grouped = segments_grouped
        else:
            self.segments = []
            self.segments_grouped = []
    
    def construct_hamiltonian(
        self,
        event: "Event | StateEventGenerator",
        convolution: bool = False,
        erf_sigma: Optional[float] = None,
        prune: "bool | dict" = False,
    ) -> tuple[csc_matrix, np.ndarray]:
        """
        Construct the Hamiltonian using vectorized operations.
        
        Parameters
        ----------
        event : Event or StateEventGenerator
            The event containing hits and geometry.
        convolution : bool, default False
            If True, use ERF-smoothed compatibility.
        erf_sigma : float or None, default None
            Override ERF smoothing width for this call.  When *None*,
            ``self.theta_d`` is used.  Passing a value here does **not**
            mutate ``self.theta_d``.
        prune : bool or dict, default False
            If true, drop the decoupled segments after the build
            (:meth:`~lhcb_velo_toy.solvers.hamiltonians.base.Hamiltonian.prune`);
            a dict is passed as keyword arguments (``min_degree``,
            ``min_component_size``).  The returned ``(A, b)`` are then the
            pruned system and ``self.kept_index`` maps back to the full
            segment index space.
        
        Returns
        -------
        tuple[csc_matrix, ndarray]
            The matrix A and vector b.
        
        Notes
        -----
        Uses COO matrix format for efficient batch construction,
        then converts to CSC for solving.
        """
        from scipy.sparse import coo_matrix as _coo
        from scipy.special import erf
        from scipy.spatial import cKDTree

        if not self._group_boundaries:
            # Preserve historical API: when callers invoke construct_hamiltonian
            # without first calling construct_segments themselves, also build
            # the eager Segment object list so ham.segments stays populated for
            # downstream truth/visualisation code.  Callers that don't need
            # Segment objects should call
            # construct_segments(event, materialize_segments=False) explicitly.
            self.construct_segments(event, materialize_segments=True)

        n = self.n_segments
        diag_val = -(self.delta + self.gamma)

        _td = erf_sigma if erf_sigma is not None else self.theta_d
        sqrt2_td = _td * np.sqrt(2)

        seg_hit_arr = np.asarray(self._segment_to_hit_ids, dtype=np.int64)
        start_ids = seg_hit_arr[:, 0]
        end_ids = seg_hit_arr[:, 1]
        seg_vecs = self._segment_vectors  # (n, 3)

        # We process each adjacent group pair (g, g+1), and within it, each
        # middle hit h.  Segments in group g ending at h pair with segments
        # in group g+1 starting at h.  Per middle hit we do an
        # (n_in_h x n_out_h) outer dot, threshold by epsilon, and append the
        # surviving (i, j, value) triplets.  Memory stays bounded by the
        # per-middle-hit block instead of the full Cartesian product.
        ii_chunks: list[np.ndarray] = []
        jj_chunks: list[np.ndarray] = []
        vv_chunks: list[np.ndarray] = []

        eps = float(self.epsilon)
        cos_thresh = float(np.cos(eps))  # angle < eps  <=>  cos > cos(eps)
        # Step path acceleration: pairs survive iff |v_i - v_j| < step_radius
        # (unit dirs => |v_i - v_j|^2 = 2(1 - cos angle); angle < eps <=> dist < r).
        # A cKDTree fixed-radius query finds the candidates without the full
        # (m x p) dot, then the SAME cos > cos_thresh test selects them, so A is
        # bit-identical to the dense scan (validated max|dA|=0, ~8.5x at T=1000).
        step_radius = float(np.sqrt(max(2.0 * (1.0 - cos_thresh), 0.0)))
        # Below this block size the dense (m x p) dot is cheaper than building a tree.
        DENSE_BLOCK = 4096
        # ERF (soft) coupling prune threshold. 1 + erf((eps-angle)/sqrt2_td)
        # decays to 0 as angle grows past eps; erf saturates by z ~ -4.5, so
        # dropping couplings below this removes only <1e-9 contributions while
        # keeping A sparse. Without it the ERF branch stores the full
        # (m x p) middle-hit block -> O(T^3) explicit near-zero entries -> OOM.
        ERF_VAL_EPS = 1e-9
        bnd = self._group_boundaries
        # bnd has len = n_modules; adjacent group pairs need bnd[g+2], so iterate to len(bnd)-2.
        for g in range(len(bnd) - 2):
            b1, b2, b3 = bnd[g], bnd[g + 1], bnd[g + 2]
            if b2 == b1 or b3 == b2:
                continue

            # Sort incoming (group g) by end_hit, outgoing (group g+1) by start_hit.
            in_idx_g = np.arange(b1, b2, dtype=np.int64)
            out_idx_g = np.arange(b2, b3, dtype=np.int64)
            in_key = end_ids[b1:b2]
            out_key = start_ids[b2:b3]

            in_order = np.argsort(in_key, kind="stable")
            in_idx_s = in_idx_g[in_order]
            in_key_s = in_key[in_order]
            out_order = np.argsort(out_key, kind="stable")
            out_idx_s = out_idx_g[out_order]
            out_key_s = out_key[out_order]

            in_unique, in_first = np.unique(in_key_s, return_index=True)
            out_unique, out_first = np.unique(out_key_s, return_index=True)
            in_first_ext = np.append(in_first, in_key_s.size)
            out_first_ext = np.append(out_first, out_key_s.size)

            # Match middle hits common to both
            common, ci, co = np.intersect1d(
                in_unique, out_unique, assume_unique=True, return_indices=True
            )

            for k in range(common.size):
                ia, ib = in_first_ext[ci[k]], in_first_ext[ci[k] + 1]
                oa, ob = out_first_ext[co[k]], out_first_ext[co[k] + 1]
                ii_chunk = in_idx_s[ia:ib]
                jj_chunk = out_idx_s[oa:ob]
                vi = seg_vecs[ii_chunk]   # (m, 3)
                vj = seg_vecs[jj_chunk]   # (p, 3)
                if convolution:
                    # ERF (soft) path: needs the value at every surviving pair, so
                    # keep the dense (m x p) block (already pruned to val>1e-9 below).
                    cos_mat = vi @ vj.T       # (m, p)
                    np.clip(cos_mat, -1.0, 1.0, out=cos_mat)
                    angle = np.arccos(cos_mat)
                    val = 1.0 + erf((eps - angle) / sqrt2_td)
                    # Keep only couplings above ERF_VAL_EPS so A stays sparse.
                    # Writing the full (m x p) block stores O(T^3) explicit
                    # near-zero entries and OOMs at high T; the soft transition
                    # near eps is preserved exactly (dropped values are <1e-9).
                    mask = val > ERF_VAL_EPS
                    if not mask.any():
                        continue
                    sel_i, sel_j = np.nonzero(mask)
                    ii_chunks.append(ii_chunk[sel_i])
                    jj_chunks.append(jj_chunk[sel_j])
                    vv_chunks.append(val[sel_i, sel_j])
                elif vi.shape[0] * vj.shape[0] <= DENSE_BLOCK:
                    # Step path, small block: dense dot is cheaper than a tree.
                    cos_mat = vi @ vj.T
                    mask = cos_mat > cos_thresh
                    if not mask.any():
                        continue
                    sel_i, sel_j = np.nonzero(mask)
                    ii_chunks.append(ii_chunk[sel_i])
                    jj_chunks.append(jj_chunk[sel_j])
                    vv_chunks.append(np.ones(sel_i.size, dtype=np.float64))
                else:
                    # Step path, large block: O((m+p) log) fixed-radius query
                    # instead of the O(m*p) dot. Re-check with the IDENTICAL
                    # cos > cos_thresh test so the surviving set is exact.
                    nbrs = cKDTree(vi).query_ball_point(vj, step_radius)
                    si_l: list[np.ndarray] = []
                    sj_l: list[np.ndarray] = []
                    for jo, lst in enumerate(nbrs):
                        if not lst:
                            continue
                        li = np.asarray(lst, dtype=np.int64)
                        keep = (vi[li] @ vj[jo]) > cos_thresh
                        if keep.any():
                            si_l.append(li[keep])
                            sj_l.append(np.full(int(keep.sum()), jo, dtype=np.int64))
                    if not si_l:
                        continue
                    sel_i = np.concatenate(si_l)
                    sel_j = np.concatenate(sj_l)
                    ii_chunks.append(ii_chunk[sel_i])
                    jj_chunks.append(jj_chunk[sel_j])
                    vv_chunks.append(np.ones(sel_i.size, dtype=np.float64))

        if ii_chunks:
            ii = np.concatenate(ii_chunks)
            jj = np.concatenate(jj_chunks)
            vv = np.concatenate(vv_chunks)
        else:
            ii = np.empty(0, dtype=np.int64)
            jj = np.empty(0, dtype=np.int64)
            vv = np.empty(0, dtype=np.float64)

        diag_rows = np.arange(n, dtype=np.int64)
        rows = np.concatenate([diag_rows, ii, jj])
        cols = np.concatenate([diag_rows, jj, ii])
        data = np.concatenate(
            [np.full(n, diag_val, dtype=np.float64), vv, vv]
        )

        A = _coo((data, (rows, cols)), shape=(n, n)).tocsc()
        b = np.ones(n) * self.delta

        self.A, self.b = -A, b
        self._maybe_prune(prune)
        return self.A, self.b
    
    def _build_hit_to_segments_map(self) -> dict[int, list[int]]:
        """
        Build mapping from hit IDs to segment indices.
        
        Returns
        -------
        dict[int, list[int]]
            Dictionary mapping hit_id to list of segment indices
            that contain that hit.
        """
        hit_to_segs: dict[int, list[int]] = {}
        for idx, (from_id, to_id) in enumerate(self._segment_to_hit_ids):
            hit_to_segs.setdefault(from_id, []).append(idx)
            hit_to_segs.setdefault(to_id, []).append(idx)
        return hit_to_segs
    
    def _compute_angles_vectorized(
        self,
        seg_indices_i: np.ndarray,
        seg_indices_j: np.ndarray,
    ) -> np.ndarray:
        """
        Compute angles between pairs of segments vectorized.
        
        Parameters
        ----------
        seg_indices_i : ndarray
            Array of first segment indices.
        seg_indices_j : ndarray
            Array of second segment indices.
        
        Returns
        -------
        ndarray
            Array of angles (in radians) between each pair.
        """
        v_i = self._segment_vectors[seg_indices_i]  # (N, 3)
        v_j = self._segment_vectors[seg_indices_j]  # (N, 3)
        dot = np.sum(v_i * v_j, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        return np.arccos(dot)
    
    def solve_classicaly(self) -> np.ndarray:
        """
        Solve the linear system with automatic solver selection.
        
        For small systems (<5000 segments), uses direct solve.
        For larger systems, uses iterative conjugate gradient.
        
        Returns
        -------
        numpy.ndarray
            Solution vector.
        
        Raises
        ------
        ValueError
            If construct_hamiltonian has not been called.
        """
        if self.A is None or self.b is None:
            raise ValueError(
                "Hamiltonian not constructed. "
                "Call construct_hamiltonian() first."
            )
        if self.n_segments < 5000:
            try:
                solution = spsolve(self.A, self.b)
            except Exception:
                solution, _ = cg(self.A, self.b, atol=1e-10)
        else:
            solution, _ = cg(self.A, self.b, atol=1e-10)
        return solution
    
    def evaluate(self, solution: np.ndarray) -> float:
        """
        Evaluate the Hamiltonian energy for a solution.
        
        Parameters
        ----------
        solution : numpy.ndarray
            Segment activation vector.
        
        Returns
        -------
        float
            Hamiltonian energy.
        """
        if self.A is None or self.b is None:
            raise ValueError(
                "Hamiltonian not constructed. "
                "Call construct_hamiltonian() first."
            )

        if isinstance(solution, list):
            sol = np.array(solution)
        else:
            sol = solution

        if sol.ndim == 1:
            sol = sol[:, np.newaxis]

        return float(-0.5 * sol.T @ self.A @ sol + self.b.dot(sol.flatten()))
