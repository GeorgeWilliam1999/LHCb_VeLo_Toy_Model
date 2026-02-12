"""
Abstract base class for event generators.

All event generators in the package inherit from :class:`EventGenerator`,
which defines the minimal interface required by the downstream solvers and
analysis pipeline.

.. todo:: LHCb Monte-Carlo import
   Implement a concrete subclass (e.g. ``MCEventLoader``) that reads
   real or simulated LHCb VELO data — for example, tracks exported from
   the LHCb Monte-Carlo (Gauss/Boole) or from the Allen
   reconstruction output — and wraps them in the same ``Event`` objects
   used by the toy generators.  This would allow the full solver and
   validation chain to run on realistic data without modification.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from lhcb_velo_toy.core.types import Position
from lhcb_velo_toy.generation.geometry.base import Geometry
from lhcb_velo_toy.generation.entities.event import Event


class EventGenerator(ABC):
    """
    Abstract base class for event generators.

    Every generator must accept a detector geometry and be able to
    produce a complete :class:`Event` via :meth:`generate_complete_events`.

    Parameters
    ----------
    detector_geometry : Geometry
        The detector geometry configuration.

    Attributes
    ----------
    detector_geometry : Geometry
        The detector geometry.
    primary_vertices : list[tuple[float, float, float]]
        Primary vertex positions.
    true_event : Event | None
        The last generated truth event (populated after generation).
    """

    def __init__(self, detector_geometry: Geometry) -> None:
        self.detector_geometry = detector_geometry
        self.primary_vertices: list[Position] = []
        self.true_event: Optional[Event] = None

    # ── required interface ──────────────────────────────────────────

    @abstractmethod
    def generate_complete_events(self) -> Event:
        """
        Generate a complete event with tracks, hits, and modules.

        Returns
        -------
        Event
            Fully populated event object.
        """
        ...

    @abstractmethod
    def make_noisy_event(
        self,
        drop_rate: float = 0.0,
        ghost_rate: float = 0.0,
    ) -> Event:
        """
        Create a noisy copy of the last generated event.

        Parameters
        ----------
        drop_rate : float
            Fraction of hits to randomly drop.
        ghost_rate : float
            Fraction of ghost hits to add.

        Returns
        -------
        Event
            New event with noise applied.
        """
        ...
