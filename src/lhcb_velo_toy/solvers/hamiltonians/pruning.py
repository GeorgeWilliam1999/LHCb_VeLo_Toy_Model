"""Pruning of decoupled segments from the segment Hamiltonian.

Provenance
----------
Added 2026-10-13 on a suggestion relayed by George: the Hamiltonian is built
from segments and every coupling is an acceptance-angle test between two
segments that share a hit, so a segment with no compatible neighbour on
either side can never be part of a reconstructed track and need not be in
the matrix at all.  The study that tests the idea is
``Quantum_Track_Reconstruction/QSVT/Segment_pruning/``.

Why it is exact
---------------
``A = s I - C`` is block-diagonal over the connected components of the
coupling graph ``C`` (segments are vertices, compatible head-to-tail pairs are
edges).  A segment with no edge is a 1 x 1 block ``[s]``.  Every solver in
this package is a matrix function of ``A`` applied to the uniform vector
``b``: the classical inverse ``A^{-1} b``, the one-bit filter's
``cos(At/2)`` and ``sin(At)``, and the QSVT polynomial ``p(A) b``.  A matrix
function of a block-diagonal matrix acts block by block, so deleting whole
blocks leaves the solution on the remaining segments unchanged.  What changes
is only

* the dimension ``n`` and therefore the system register ``ceil(log2 n)``,
* the global normalisation of a quantum state (the post-selection success
  probability rises by the fraction of the register that was decoupled),
* the number of controls on every multi-controlled gate of the one-bit
  circuit (one per system qubit).

The decoupled segments are the "isolated" segments of the error atlas: the
classical solver puts them at the attractor ``delta / s`` and the one-bit
filter sends them to exactly zero, so no solver ever reads them as track
segments.  A *true* segment can be decoupled only when its track lost a hit
on both sides (hit inefficiency); pruning removes it, and the study reports
that count separately because no segment-pair solver can recover it either.

The same argument removes any union of whole components, which is what
``min_component_size`` does (a lone compatible pair, for example, is never a
track of three or more hits on a five-plane detector); that is a stronger,
still exact, prune whose effect on the metrics is a design choice rather than
a free lunch, so the default is the weakest prune: degree zero only.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

__all__ = [
    "coupling_pattern",
    "segment_degree",
    "decoupled_mask",
    "prune_system",
    "expand_solution",
    "qubits_for",
]


def qubits_for(n: int) -> int:
    """System-register width for ``n`` segments: ``max(1, ceil(log2 n))``."""
    return max(1, int(math.ceil(math.log2(max(int(n), 1))))) if n > 1 else 1


def coupling_pattern(A) -> sp.csr_matrix:
    """0/1 pattern of the stored off-diagonal entries of ``A`` (CSR).

    Explicit zeros are dropped, so an entry counts as a coupling only when
    the builder actually stored a non-zero value (for the ERF kernel that is
    a value above its own prune floor).
    """
    C = (A.tocsr() if sp.issparse(A) else sp.csr_matrix(np.asarray(A, float))).copy()
    C.setdiag(0.0)
    C.eliminate_zeros()
    C.data = np.ones_like(C.data)
    return C


def segment_degree(A) -> np.ndarray:
    """Number of compatible neighbours of every segment (off-diagonal nnz per row)."""
    return np.asarray(coupling_pattern(A).getnnz(axis=1)).ravel().astype(np.int64)


def decoupled_mask(A, min_degree: int = 1, min_component_size: int = 1):
    """Boolean keep-mask over the segments of ``A`` and a diagnostics dict.

    Parameters
    ----------
    A : (n, n) sparse Hermitian
        The segment Hamiltonian (any builder; only the off-diagonal pattern
        is read).
    min_degree : int, default 1
        Keep a segment only if it has at least this many compatible
        neighbours.  ``1`` is the decoupled-segment prune; ``0`` keeps all.
    min_component_size : int, default 1
        Keep a segment only if its connected component of the coupling graph
        has at least this many segments.  ``1`` imposes nothing; ``3`` drops
        lone compatible pairs as well as singletons.

    Returns
    -------
    keep : (n,) bool
    info : dict
        ``n_full, n_kept, n_pruned, min_degree, min_component_size,
        n_components, max_component, n_singletons, qubits_full, qubits_kept``.
    """
    C = coupling_pattern(A)
    n = C.shape[0]
    deg = np.asarray(C.getnnz(axis=1)).ravel()
    keep = deg >= int(min_degree)
    n_comp, lab = connected_components(C, directed=False)
    sizes = np.bincount(lab, minlength=n_comp)
    if int(min_component_size) > 1:
        keep &= sizes[lab] >= int(min_component_size)
    n_kept = int(keep.sum())
    info = dict(
        n_full=int(n),
        n_kept=n_kept,
        n_pruned=int(n - n_kept),
        min_degree=int(min_degree),
        min_component_size=int(min_component_size),
        n_components=int(n_comp),
        max_component=int(sizes.max()) if n_comp else 0,
        n_singletons=int((sizes == 1).sum()),
        qubits_full=qubits_for(n),
        qubits_kept=qubits_for(n_kept),
    )
    return keep, info


def prune_system(A, b, keep):
    """Restrict ``(A, b)`` to the segments where ``keep`` is True.

    Returns ``(A_kept, b_kept, kept_index)`` with ``A_kept`` in CSC form
    (the builders' convention) and ``kept_index`` the original indices.
    """
    keep = np.asarray(keep, bool)
    kept = np.nonzero(keep)[0]
    A_csr = A.tocsr() if sp.issparse(A) else sp.csr_matrix(np.asarray(A, float))
    A_k = A_csr[kept][:, kept].tocsc()
    b_k = np.asarray(b).ravel()[kept]
    return A_k, b_k, kept


def expand_solution(vec, kept_index, n_full: int, fill: float = 0.0) -> np.ndarray:
    """Scatter a kept-space vector back to the full segment index space.

    Pruned positions receive ``fill`` (0 by default: the one-bit filter's
    own value for a decoupled segment; pass ``delta / s`` to mimic the
    classical attractor instead).
    """
    vec = np.asarray(vec)
    out = np.full(int(n_full), fill, dtype=np.result_type(vec.dtype, np.float64))
    out[np.asarray(kept_index, np.int64)] = vec
    return out
