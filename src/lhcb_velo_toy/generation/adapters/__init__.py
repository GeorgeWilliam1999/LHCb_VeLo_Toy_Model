"""Adapters turning external event formats into toy-model-consumable events."""

from lhcb_velo_toy.generation.adapters.run3 import Run3Event, load_run3_event

__all__ = ["Run3Event", "load_run3_event"]
