"""
Abstract base class for track-finding Hamiltonians.

The Hamiltonian formulation converts track finding into an optimization
problem that can be solved classically or with quantum algorithms.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.sparse import csc_matrix

if TYPE_CHECKING:
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment
    from lhcb_velo_toy.generation.entities.event import Event
    from lhcb_velo_toy.generation.generators.state_event import StateEventGenerator


class Hamiltonian(ABC):
    """
    Abstract base class for track-finding Hamiltonians.
    
    A Hamiltonian encodes the track-finding problem as a quadratic
    optimization problem:
    
    .. math::
        H(\\mathbf{x}) = -\\frac{1}{2} \\mathbf{x}^T A \\mathbf{x} + \\mathbf{b}^T \\mathbf{x}
    
    where x is the segment activation vector. The minimum of H corresponds
    to the optimal track configuration.
    
    Attributes
    ----------
    A : scipy.sparse.csc_matrix
        The Hamiltonian matrix encoding segment interactions.
    b : numpy.ndarray
        The bias vector.
    segments : list[Segment]
        List of all segment candidates.
    n_segments : int
        Total number of segments.
    
    Notes
    -----
    Subclasses must implement:
    - `construct_hamiltonian`: Build the A matrix and b vector
    - `evaluate`: Compute H(x) for a given solution
    
    The matrix A typically encodes:
    - Diagonal: Self-interaction penalties (discouraging segment activation)
    - Off-diagonal: Compatibility between segments (angular alignment)
    """
    
    A: Optional[csc_matrix] = None
    b: Optional[np.ndarray] = None
    segments: list["Segment"]
    n_segments: int
    # Pruning state (see :meth:`prune`): ``None`` until a prune is applied.
    kept_index: Optional[np.ndarray] = None
    n_segments_full: Optional[int] = None
    prune_info: dict = {}
    
    @abstractmethod
    def construct_hamiltonian(
        self,
        event: "Event | StateEventGenerator",
        convolution: bool = False,
    ) -> tuple[csc_matrix, np.ndarray]:
        """
        Construct the Hamiltonian matrix and bias vector.
        
        Parameters
        ----------
        event : Event or StateEventGenerator
            The event containing hits and geometry information.
        convolution : bool, default False
            If True, use ERF-smoothed thresholding for angular compatibility.
            If False, use hard threshold.
        
        Returns
        -------
        tuple[csc_matrix, ndarray]
            The Hamiltonian matrix A and bias vector b.
        
        Notes
        -----
        This method should:
        1. Construct all segment candidates from adjacent module hits
        2. Compute angular compatibility between segments
        3. Build the sparse matrix A with:
           - Diagonal: -(γ + δ)
           - Off-diagonal: 1 if segments share a hit and are compatible
        4. Build the bias vector b with entries (γ + δ)
        """
        ...
    
    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> float:
        """
        Evaluate the Hamiltonian for a given solution.
        
        Computes H(x) = -0.5 * x^T A x + b^T x
        
        Parameters
        ----------
        solution : numpy.ndarray
            Segment activation vector of length n_segments.
        
        Returns
        -------
        float
            The Hamiltonian energy value.
        
        Raises
        ------
        ValueError
            If solution length doesn't match n_segments.
        """
        ...
    
    def solve_classicaly(self) -> np.ndarray:
        """
        Solve the linear system A x = b classically.
        
        Uses iterative conjugate gradient or direct methods depending
        on matrix size and properties.
        
        Returns
        -------
        numpy.ndarray
            Solution vector x.
        
        Raises
        ------
        ValueError
            If construct_hamiltonian has not been called.
        """
        from scipy.sparse.linalg import cg

        if self.A is None or self.b is None:
            raise ValueError(
                "Hamiltonian not constructed. "
                "Call construct_hamiltonian() first."
            )
        solution, _ = cg(self.A, self.b, atol=0)
        return solution
    
    # ------------------------------------------------------------------
    # Pruning of decoupled segments (2026-10-13)
    # ------------------------------------------------------------------
    _PER_SEGMENT_ARRAYS = (
        "_segment_vectors",
        "_segment_to_hit_ids",
        "_segment_track_ids",
        "_segment_endpoints_xyz",
    )

    def prune(self, min_degree: int = 1, min_component_size: int = 1):
        """Drop the segments that cannot belong to any track from ``A``/``b``.

        A segment with no compatible neighbour (no off-diagonal entry in its
        row) is a 1 x 1 block of the block-diagonal operator; every solver is
        a matrix function of ``A`` acting block by block, so removing it
        leaves the solution on the remaining segments unchanged (module
        docstring of :mod:`lhcb_velo_toy.solvers.hamiltonians.pruning`).

        After the call ``self.A``, ``self.b``, ``self.n_segments``,
        ``self.segments`` and the cached per-segment arrays all refer to the
        kept segments only, in their original relative order.  The original
        index of kept segment ``k`` is ``self.kept_index[k]``; the full
        per-segment arrays are kept under ``<name>_full`` and the full list
        of ``Segment`` objects (if it was materialised) under
        ``self.segments_full``.  :meth:`expand` scatters a kept-space
        solution back to the full index space.

        Parameters
        ----------
        min_degree : int, default 1
            Keep segments with at least this many compatible neighbours.
        min_component_size : int, default 1
            Keep segments whose connected component has at least this many
            segments (a stronger, still exact, prune; ``1`` imposes nothing).

        Returns
        -------
        (kept_index, info) : the original indices of the kept segments and
        the diagnostics dict of
        :func:`~lhcb_velo_toy.solvers.hamiltonians.pruning.decoupled_mask`.

        Raises
        ------
        ValueError
            If the Hamiltonian is not built or has already been pruned
            (compose the criteria in one call instead).
        """
        from lhcb_velo_toy.solvers.hamiltonians.pruning import (
            decoupled_mask,
            prune_system,
        )

        if self.A is None or self.b is None:
            raise ValueError(
                "Hamiltonian not constructed. "
                "Call construct_hamiltonian() first."
            )
        if self.kept_index is not None:
            raise ValueError(
                "Hamiltonian already pruned; rebuild it before pruning again."
            )
        n_full = int(self.A.shape[0])
        keep, info = decoupled_mask(
            self.A, min_degree=min_degree, min_component_size=min_component_size
        )
        A_k, b_k, kept = prune_system(self.A, self.b, keep)

        for name in self._PER_SEGMENT_ARRAYS:
            arr = getattr(self, name, None)
            if arr is not None and len(arr) == n_full:
                setattr(self, name + "_full", arr)
                setattr(self, name, np.asarray(arr)[kept])
        segs = getattr(self, "segments", None)
        if segs:
            self.segments_full = segs
            self.segments = [segs[i] for i in kept]
            keep_set = set(kept.tolist())
            grouped = getattr(self, "segments_grouped", None)
            if grouped:
                self.segments_grouped = [
                    [sg for sg in grp if sg.segment_id in keep_set]
                    for grp in grouped
                ]
        # The module offsets no longer describe the kept arrays; a rebuild
        # must start from the segments again.
        if hasattr(self, "_group_boundaries"):
            self._group_boundaries = []

        self.A, self.b = A_k, b_k
        self.n_segments_full = n_full
        self.n_segments = int(len(kept))
        self.kept_index = kept
        self.prune_info = dict(info)
        return kept, self.prune_info

    def expand(self, vec, fill: float = 0.0) -> np.ndarray:
        """Scatter a kept-space vector to the full segment index space.

        Identity when the Hamiltonian was not pruned.  Pruned positions get
        ``fill`` (0 by default, the one-bit filter's value for a decoupled
        segment).
        """
        from lhcb_velo_toy.solvers.hamiltonians.pruning import expand_solution

        if self.kept_index is None:
            return np.asarray(vec)
        return expand_solution(vec, self.kept_index, self.n_segments_full, fill)

    def _maybe_prune(self, prune):
        """Apply ``prune`` as given to ``construct_hamiltonian`` (bool or dict)."""
        if not prune:
            return
        opts = dict(prune) if isinstance(prune, dict) else {}
        self.prune(**opts)

    def get_matrix_dense(self) -> np.ndarray:
        """
        Get the Hamiltonian matrix as a dense array.
        
        Returns
        -------
        numpy.ndarray
            Dense representation of A.
        
        Raises
        ------
        ValueError
            If construct_hamiltonian has not been called.
        
        Warnings
        --------
        This can consume significant memory for large matrices.
        """
        if self.A is None:
            raise ValueError(
                "Hamiltonian not constructed. "
                "Call construct_hamiltonian() first."
            )
        return self.A.toarray()
