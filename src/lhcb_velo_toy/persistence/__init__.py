"""
Persistence utilities for the LHCb VELO Toy Model.

Provides save / load functions for every stage of the reconstruction
pipeline and for parametric sweep studies, so that expensive
computations (event generation, Hamiltonian construction, solving,
validation) need not be repeated between notebook sessions.

Submodules
----------
pipeline
    Single-event pipeline state (event, Hamiltonian, solution,
    validation).
study
    Parametric sweep results (scan arrays, histogram distributions).

Quick-start
-----------
>>> from lhcb_velo_toy.persistence import (
...     save_pipeline, load_pipeline,
...     save_events_batch, load_events_batch,
...     save_study, load_study,
... )
"""

from lhcb_velo_toy.persistence.pipeline import (
    PipelineResult,
    load_events_batch,
    load_pipeline,
    save_events_batch,
    save_pipeline,
)
from lhcb_velo_toy.persistence.study import (
    StudyResult,
    load_study,
    save_study,
)

__all__ = [
    # pipeline
    "PipelineResult",
    "save_pipeline",
    "load_pipeline",
    "save_events_batch",
    "load_events_batch",
    # study
    "StudyResult",
    "save_study",
    "load_study",
]
