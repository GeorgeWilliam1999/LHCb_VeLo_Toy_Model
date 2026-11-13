"""Primary-vertex finding.

pv_sampler
    Direct PV readout by an amplitude-amplified vertex sampler (quantum), with its
    exact output distribution, qubits and queries, and the classical counterparts.
track_vertices
    PVs from first-pass tracks: z clustering plus a 3D vertex fit (classical baseline).
"""
from lhcb_velo_toy.solvers.vertexing.pv_sampler import (  # noqa: F401
    VertexOracleTable,
    amplification,
    doublet_table,
    find_vertex_peaks,
    fit_line_poca,
    forward_couplings,
    line_poca,
    match_vertices,
    output_distribution,
    sample_counts,
    sampler_resources,
    triplet_table,
    workspace_qubits_model,
    zeta_bins,
)
from lhcb_velo_toy.solvers.vertexing.track_vertices import (  # noqa: F401
    fit_track_lines,
    fit_vertex_point,
    vertices_from_tracks,
)
