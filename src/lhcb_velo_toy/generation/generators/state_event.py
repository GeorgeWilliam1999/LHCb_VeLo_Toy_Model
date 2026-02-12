"""
StateEventGenerator: Primary event generator using LHCb state vectors.

This generator creates collision events by propagating particles through
the detector using LHCb-style state vectors (x, y, tx, ty, p/q).

Physics Notes
-------------
Two distinct noise sources are modelled during event generation:

1. **Multiple scattering** (``collision_noise``):
   A *real* physical process. As a charged particle traverses detector
   material it undergoes many small-angle Coulomb scatters off atomic
   nuclei.  This genuinely alters the particle's direction of travel
   (slopes tx, ty).  It is applied to the **true particle state** after
   a hit is recorded, so that propagation to the *next* module uses the
   updated (scattered) trajectory.

2. **Measurement error** (``measurement_error``):
   A *detector artefact*.  The particle's true (x, y) position at the
   sensor is smeared by finite spatial resolution when it is recorded as
   a Hit.  Crucially, this noise is applied **only to the stored Hit
   coordinates** and does **not** feed back into the true particle state.
   The particle continues to propagate from its true position.

Correct ordering at each detector module:

    1. Propagate true state to module z   → true (x, y)
    2. Check if true (x, y) is on the active bulk
    3. Record Hit at (x + σ_meas, y + σ_meas)   ← measurement error
    4. Apply multiple scattering to true (tx, ty) ← real physics
    5. Proceed to next module

This ensures measurement noise never contaminates the true trajectory,
while scattering correctly accumulates through the detector.

.. todo:: Bidirectional propagation
   Currently particles are propagated strictly forward (increasing z)
   through the detector modules. In the real LHCb VELO the primary
   vertex can occur *inside* the detector acceptance — not necessarily
   before the first module. Particles should therefore propagate both
   forward and backward from the PV, hitting modules on either side.
   Investigate adapting ``generate_complete_events`` to sort modules
   relative to PV z and propagate in both directions.
"""

from __future__ import annotations
from itertools import count
from typing import Optional

import numpy as np

from lhcb_velo_toy.core.types import StateVector, Position
from lhcb_velo_toy.generation.geometry.base import Geometry
from lhcb_velo_toy.generation.generators.base import EventGenerator
from lhcb_velo_toy.generation.entities.event import Event
from lhcb_velo_toy.generation.entities.hit import Hit
from lhcb_velo_toy.generation.entities.track import Track
from lhcb_velo_toy.generation.entities.module import Module
from lhcb_velo_toy.generation.entities.primary_vertex import PrimaryVertex


class StateEventGenerator(EventGenerator):
    """
    Event generator using LHCb state vectors.

    This class generates simulated collision events by creating particles
    at primary vertices and propagating them through the detector geometry.
    Particles are represented by LHCb-style state vectors containing position
    and slope information.

    Parameters
    ----------
    detector_geometry : Geometry
        The detector geometry configuration.
    primary_vertices : list[tuple[float, float, float]], optional
        Pre-defined primary vertex positions as (x, y, z) tuples.
    phi_min : float, default -0.2
        Minimum angle for tx slope (radians).
    phi_max : float, default 0.2
        Maximum angle for tx slope (radians).
    theta_min : float, default -0.2
        Minimum angle for ty slope (radians).
    theta_max : float, default 0.2
        Maximum angle for ty slope (radians).
    events : int, default 3
        Number of collision events to generate.
    n_particles : list[int], optional
        Number of particles per event.
    measurement_error : float, default 0.0
        Gaussian σ for hit position smearing (mm).  Applied only to the
        *recorded* hit coordinates; the true particle state is unaffected.
    collision_noise : float, default 1e-4
        Gaussian σ for multiple scattering angle (radians).  Applied to
        the true particle slopes (tx, ty) after each module crossing.

    Attributes
    ----------
    detector_geometry : Geometry
        The detector geometry.
    primary_vertices : list[tuple]
        Primary vertex positions.
    particles : list[list[dict]]
        Generated particle state vectors (per-event, per-particle).
    true_event : Event
        The last generated truth event.
    hits : list[Hit]
        All hits from the last generated event.
    tracks : list[Track]
        All tracks from the last generated event.
    modules : list[Module]
        Detector modules with assigned hits.

    Examples
    --------
    >>> geometry = PlaneGeometry(
    ...     module_id=[1, 2, 3, 4, 5],
    ...     lx=[33.0] * 5, ly=[33.0] * 5,
    ...     z=[20.0, 40.0, 60.0, 80.0, 100.0]
    ... )
    >>> gen = StateEventGenerator(
    ...     detector_geometry=geometry,
    ...     events=1,
    ...     n_particles=[10],
    ...     measurement_error=0.005
    ... )
    >>> gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 1.0})
    >>> gen.generate_particles([[{"type": "MIP", "mass": 0.511, "q": 1}
    ...                          for _ in range(10)]])
    >>> event = gen.generate_complete_events()

    Notes
    -----
    The state vector uses LHCb conventions:
    - x, y, z: position in mm
    - tx = dx/dz, ty = dy/dz: slopes
    - p/q: momentum divided by charge (MeV/c)

    **Important:** Measurement error and multiple scattering are distinct:

    - *Measurement error* smears only the **recorded Hit** position.
      The true particle state is not modified.
    - *Multiple scattering* modifies the **true particle slopes** (tx, ty).
      This is a real physical deflection that accumulates through the
      detector.
    """

    def __init__(
        self,
        detector_geometry: Geometry,
        primary_vertices: Optional[list[Position]] = None,
        phi_min: float = -0.2,
        phi_max: float = 0.2,
        theta_min: float = -0.2,
        theta_max: float = 0.2,
        events: int = 3,
        n_particles: Optional[list[int]] = None,
        particles: Optional[list[list[dict]]] = None,
        measurement_error: float = 0.0,
        collision_noise: float = 1e-4,
    ) -> None:
        """Initialize the event generator."""
        super().__init__(detector_geometry)
        self.primary_vertices: list[Position] = (
            primary_vertices if primary_vertices is not None else []
        )
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.events_num = events
        self.n_particles: list[int] = (
            n_particles if n_particles is not None else []
        )
        self.particles: list[list[dict]] = (
            particles if particles is not None else []
        )
        self.measurement_error = measurement_error
        self.collision_noise = collision_noise

        self.rng = np.random.default_rng()

        # Populated after generate_complete_events()
        self.hits: list[Hit] = []
        self.tracks: list[Track] = []
        self.modules: list[Module] = []
        self.true_event: Optional[Event] = None

    # =========================================================================
    # Primary Vertex Generation
    # =========================================================================

    def generate_random_primary_vertices(
        self,
        variance: dict[str, float],
    ) -> list[Position]:
        """
        Generate Gaussian-distributed primary vertices.

        Creates primary vertex positions for each event, sampled from
        Gaussian distributions centred at the origin.

        Parameters
        ----------
        variance : dict[str, float]
            Dictionary with keys from {'x', 'y', 'z'} specifying the
            standard deviation (σ) for each coordinate.  Missing keys
            default to 0.

        Returns
        -------
        list[tuple[float, float, float]]
            List of (x, y, z) vertex positions, one per event.

        Examples
        --------
        >>> gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 1.0})
        [(0.05, -0.03, 0.8), ...]
        """
        primary_vertices: list[Position] = []
        sigma_x = variance.get("x", 0.0)
        sigma_y = variance.get("y", 0.0)
        sigma_z = variance.get("z", 0.0)

        for _ in range(self.events_num):
            x = self.rng.normal(0.0, sigma_x) if sigma_x > 0 else 0.0
            y = self.rng.normal(0.0, sigma_y) if sigma_y > 0 else 0.0
            z = self.rng.normal(0.0, sigma_z) if sigma_z > 0 else 0.0
            primary_vertices.append((x, y, z))

        self.primary_vertices = primary_vertices
        return primary_vertices

    def set_primary_vertices(
        self,
        vertices: list[Position],
    ) -> None:
        """
        Set explicit primary vertex positions.

        Parameters
        ----------
        vertices : list[tuple[float, float, float]]
            List of (x, y, z) vertex positions.  Must have length equal
            to the number of events.

        Raises
        ------
        ValueError
            If the number of vertices doesn't match the number of events.
        ValueError
            If any vertex tuple does not have exactly 3 elements.
        """
        if len(vertices) != self.events_num:
            raise ValueError(
                f"Expected {self.events_num} vertices, got {len(vertices)}"
            )
        for v in vertices:
            if len(v) != 3:
                raise ValueError(
                    f"Each vertex must be a 3-tuple (x, y, z), got length {len(v)}"
                )
        self.primary_vertices = list(vertices)

    # =========================================================================
    # Particle Generation
    # =========================================================================

    def generate_particles(
        self,
        particles: list[list[dict]],
    ) -> list[list[dict]]:
        """
        Generate particle state vectors.

        Creates initial state vectors for particles based on the provided
        particle configurations.  Slopes (tx, ty) are sampled uniformly
        from the configured angular ranges.

        Parameters
        ----------
        particles : list[list[dict]]
            Nested list of particle configurations.  Outer list is per-event,
            inner list is per-particle.  Each particle dict may contain:
            - 'type': Particle type (e.g., 'MIP')
            - 'mass': Particle mass (MeV/c²)
            - 'q':   Particle charge (default 1)

        Returns
        -------
        list[list[dict]]
            Nested list of particle state dictionaries (per-event,
            per-particle).  Each state dict has keys:
            'type', 'x', 'y', 'z', 'tx', 'ty', 'p/q'.

        Examples
        --------
        >>> particles = [[{"type": "MIP", "mass": 0.511, "q": 1}
        ...               for _ in range(5)]]
        >>> gen.generate_particles(particles)
        """
        all_particles: list[list[dict]] = []

        for evt_idx in range(self.events_num):
            vx, vy, vz = self.primary_vertices[evt_idx]
            event_particles: list[dict] = []

            for p_idx in range(self.n_particles[evt_idx]):
                # Sample angles uniformly
                phi = self.rng.uniform(self.phi_min, self.phi_max)
                theta = self.rng.uniform(self.theta_min, self.theta_max)

                tx = np.tan(phi)
                ty = np.tan(theta)

                # Extract particle properties from input config
                p_conf = particles[evt_idx][p_idx]
                q = p_conf.get("q", 1)
                mass = p_conf.get("mass", 0.511)
                p = np.linalg.norm([tx, ty, 1.0]) * mass * 0.89

                event_particles.append(
                    {
                        "type": p_conf.get("type", "MIP"),
                        "x": vx,
                        "y": vy,
                        "z": vz,
                        "tx": tx,
                        "ty": ty,
                        "p/q": p / q,
                    }
                )

            all_particles.append(event_particles)

        self.particles = all_particles
        return all_particles

    # =========================================================================
    # Propagation & Physics
    # =========================================================================

    def propagate(
        self,
        state: StateVector,
        z_target: float,
    ) -> StateVector:
        """
        Propagate a particle state to a target z position.

        Uses linear extrapolation based on the particle slopes.
        Modifies the state **in-place** and returns it.

        Parameters
        ----------
        state : dict[str, float]
            Current particle state with 'x', 'y', 'z', 'tx', 'ty'.
        z_target : float
            Target z position (mm).

        Returns
        -------
        dict[str, float]
            Updated state at the target z position.

        Notes
        -----
        Propagation equations:
        - x_new = x + tx × (z_target − z)
        - y_new = y + ty × (z_target − z)
        """
        dz = z_target - state["z"]
        state["x"] += state["tx"] * dz
        state["y"] += state["ty"] * dz
        state["z"] = z_target
        return state

    def collision_update(
        self,
        state: StateVector,
    ) -> StateVector:
        """
        Apply multiple scattering to the true particle state.

        Adds Gaussian-distributed angular deflections to the particle
        slopes (tx, ty) to simulate multiple Coulomb scattering in
        detector material.  This is a **real physical process** that
        genuinely alters the particle trajectory.

        Parameters
        ----------
        state : dict[str, float]
            Current (true) particle state.

        Returns
        -------
        dict[str, float]
            Updated state with scattered slopes.

        Notes
        -----
        The scattering is modelled as independent Gaussian kicks in
        tx and ty with σ = ``collision_noise``.  This is applied to
        the **true** particle state (not the recorded hit).
        """
        state["tx"] += np.tan(self.rng.normal(0.0, self.collision_noise))
        state["ty"] += np.tan(self.rng.normal(0.0, self.collision_noise))
        return state

    def measurement_error_update(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """
        Apply measurement error to hit positions.

        Adds Gaussian noise to simulate finite detector spatial resolution.
        This is a **detector artefact only** — the true particle state is
        unaffected.  Only the recorded Hit coordinates are smeared.

        Parameters
        ----------
        x : float
            True x coordinate (mm).
        y : float
            True y coordinate (mm).

        Returns
        -------
        tuple[float, float]
            Smeared (x_measured, y_measured) coordinates.

        Notes
        -----
        The measurement error is drawn from N(0, σ²) where
        σ = ``self.measurement_error``.  If ``measurement_error`` is 0
        the true coordinates are returned unchanged.
        """
        if self.measurement_error == 0.0:
            return x, y
        x_meas = x + self.rng.normal(0.0, self.measurement_error)
        y_meas = y + self.rng.normal(0.0, self.measurement_error)
        return x_meas, y_meas

    # =========================================================================
    # Event Generation
    # =========================================================================

    def generate_complete_events(self) -> Event:
        """
        Generate complete events with tracks and hits.

        Propagates all particles through the detector, recording hits
        at each module crossing, and constructs the full Event structure
        with PrimaryVertices, Tracks, Hits, and Modules.

        Returns
        -------
        Event
            Complete event with geometry, primary vertices, tracks, hits,
            and modules.

        Notes
        -----
        Processing order at each detector module (see module docstring):

        1. **Propagate** true state to module z → gives true (x, y)
        2. **Check** if true (x, y) is on the active bulk
        3. **Record Hit** at ``(x + σ_meas, y + σ_meas)``
           — measurement error applied only to the stored Hit
        4. **Apply multiple scattering** — modify true (tx, ty)
           for propagation to the next module
        5. Proceed to next module

        This ensures measurement noise never contaminates the true
        trajectory, while scattering correctly accumulates.
        """
        hit_counter = count()
        track_counter = count()

        all_hits: list[Hit] = []
        all_tracks: list[Track] = []
        all_pvs: list[PrimaryVertex] = []

        for evt_idx in range(self.events_num):
            vx, vy, vz = self.primary_vertices[evt_idx]

            # Create PrimaryVertex
            pv = PrimaryVertex(
                pv_id=evt_idx,
                x=vx,
                y=vy,
                z=vz,
                track_ids=[],
            )

            for p_idx in range(self.n_particles[evt_idx]):
                track_id = next(track_counter)
                track_hit_ids: list[int] = []

                # Copy the particle state so we don't mutate the stored
                # initial conditions
                state = dict(self.particles[evt_idx][p_idx])

                # Propagate through each detector module
                for mod_id, lx, ly, zpos in self.detector_geometry:
                    # 1. Propagate true state to this module
                    state = self.propagate(state, zpos)

                    # 2. Check acceptance
                    if not self.detector_geometry.point_on_bulk(state):
                        # Not on active detector area — no hit recorded and
                        # no material interaction so no scattering.  The
                        # particle may still hit a later module.
                        continue

                    # 3. Record hit (measurement error on recorded coords only)
                    hit_id = next(hit_counter)
                    x_meas, y_meas = self.measurement_error_update(
                        state["x"], state["y"]
                    )

                    hit = Hit(
                        hit_id=hit_id,
                        x=x_meas,
                        y=y_meas,
                        z=zpos,
                        module_id=mod_id,
                        track_id=track_id,
                    )
                    all_hits.append(hit)
                    track_hit_ids.append(hit_id)

                    # 4. Apply multiple scattering to TRUE state
                    state = self.collision_update(state)

                # Build track
                track = Track(
                    track_id=track_id,
                    pv_id=pv.pv_id,
                    hit_ids=track_hit_ids,
                )
                all_tracks.append(track)
                pv.add_track(track_id)

            all_pvs.append(pv)

        # Build modules with hit_ids
        modules: list[Module] = []
        for mod_id, lx, ly, zpos in self.detector_geometry:
            mod_hit_ids = [h.hit_id for h in all_hits if h.module_id == mod_id]
            modules.append(
                Module(
                    module_id=mod_id,
                    z=zpos,
                    lx=lx,
                    ly=ly,
                    hit_ids=mod_hit_ids,
                )
            )

        # Store on instance for later access / make_noisy_event
        self.hits = all_hits
        self.tracks = all_tracks
        self.modules = modules

        self.true_event = Event(
            detector_geometry=self.detector_geometry,
            primary_vertices=all_pvs,
            tracks=all_tracks,
            hits=all_hits,
            modules=modules,
        )

        return self.true_event

    # =========================================================================
    # Noise injection
    # =========================================================================

    def make_noisy_event(
        self,
        drop_rate: float = 0.0,
        ghost_rate: float = 0.0,
    ) -> Event:
        """
        Create a noisy version of the generated event.

        Simulates detector inefficiencies by randomly removing hits
        (drop-outs) and adding ghost (fake) hits.

        Parameters
        ----------
        drop_rate : float, default 0.0
            Fraction of hits to randomly drop (0 to 1).
        ghost_rate : float, default 0.0
            Fraction of extra ghost hits to add, relative to the
            original number of hits (0 to 1).

        Returns
        -------
        Event
            New Event with dropped and ghost hits applied.

        Notes
        -----
        - Dropped hits are removed from the global hit list, from their
          parent Track's ``hit_ids``, and from their Module's ``hit_ids``.
        - Ghost hits are assigned ``track_id = -1``.
        - The original ``true_event`` is not modified.

        Examples
        --------
        >>> noisy_event = gen.make_noisy_event(drop_rate=0.1, ghost_rate=0.05)
        """
        if self.true_event is None:
            raise RuntimeError(
                "Call generate_complete_events() before make_noisy_event()"
            )

        # Work on copies so the true event is preserved
        remaining_hits = list(self.hits)
        total_hits = len(remaining_hits)

        # --- Drop hits ---
        if drop_rate > 0.0:
            n_drop = int(total_hits * drop_rate)
            drop_indices = set(
                self.rng.choice(total_hits, size=n_drop, replace=False)
            )
            remaining_hits = [
                h for i, h in enumerate(remaining_hits) if i not in drop_indices
            ]

        remaining_hit_ids = {h.hit_id for h in remaining_hits}

        # Update tracks to reflect dropped hits
        noisy_tracks: list[Track] = []
        for t in self.tracks:
            new_hit_ids = [hid for hid in t.hit_ids if hid in remaining_hit_ids]
            noisy_tracks.append(
                Track(track_id=t.track_id, pv_id=t.pv_id, hit_ids=new_hit_ids)
            )

        # --- Add ghost hits ---
        ghost_id_counter = count(max((h.hit_id for h in remaining_hits), default=-1) + 1)

        if ghost_rate > 0.0:
            n_ghosts = int(total_hits * ghost_rate)
            for _ in range(n_ghosts):
                # Pick a random module
                idx = self.rng.integers(0, len(self.detector_geometry))
                mod_data = self.detector_geometry[idx]
                mod_id = mod_data[0]
                lx = mod_data[-3] if len(mod_data) > 4 else mod_data[1]
                ly = mod_data[-2] if len(mod_data) > 4 else mod_data[2]
                zpos = mod_data[-1] if len(mod_data) > 4 else mod_data[3]

                ghost_hit = Hit(
                    hit_id=next(ghost_id_counter),
                    x=self.rng.uniform(-lx, lx),
                    y=self.rng.uniform(-ly, ly),
                    z=zpos,
                    module_id=mod_id,
                    track_id=-1,
                )
                remaining_hits.append(ghost_hit)

        # Rebuild modules
        noisy_modules = self._build_modules(remaining_hits)

        # Reuse PVs from the truth event
        noisy_pvs = (
            list(self.true_event.primary_vertices)
            if self.true_event.primary_vertices
            else []
        )

        noisy_event = Event(
            detector_geometry=self.detector_geometry,
            primary_vertices=noisy_pvs,
            tracks=noisy_tracks,
            hits=remaining_hits,
            modules=noisy_modules,
        )
        return noisy_event

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _build_modules(self, hits: list[Hit]) -> list[Module]:
        """Build Module list from a flat list of hits."""
        modules: list[Module] = []
        for mod_id, lx, ly, zpos in self.detector_geometry:
            mod_hit_ids = [h.hit_id for h in hits if h.module_id == mod_id]
            modules.append(
                Module(
                    module_id=mod_id,
                    z=zpos,
                    lx=lx,
                    ly=ly,
                    hit_ids=mod_hit_ids,
                )
            )
        return modules

    def _rebuild_modules(self) -> None:
        """Rebuild module hit lists from current self.hits."""
        self.modules = self._build_modules(self.hits)
