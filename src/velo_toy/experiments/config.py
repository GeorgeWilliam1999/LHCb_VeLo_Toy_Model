"""
Experiment configuration classes.

Defines the parameter space for Hamiltonian track reconstruction experiments.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np


@dataclass
class DetectorConfig:
    """Configuration for the VELO detector geometry."""
    
    layers: int = 5
    dz_mm: float = 33.0
    lx_mm: float = 80.0
    ly_mm: float = 80.0
    
    @property
    def module_ids(self) -> List[int]:
        return list(range(1, self.layers + 1))
    
    @property
    def zs_mm(self) -> List[float]:
        return [self.dz_mm * l for l in range(1, self.layers + 1)]
    
    @property
    def lx_list(self) -> List[float]:
        return [self.lx_mm] * self.layers
    
    @property
    def ly_list(self) -> List[float]:
        return [self.ly_mm] * self.layers


# Default detector configuration
DEFAULT_DETECTOR = DetectorConfig()


@dataclass
class ExperimentConfig:
    """
    Configuration for a single track reconstruction experiment.
    
    Physics Parameters:
        meas_error: Hit position measurement error (sigma)
        coll_noise: Multiple scattering angle parameter
        ghost_rate: Fraction of ghost (fake) hits to inject
        drop_rate: Fraction of true hits to drop (inefficiency)
    
    Hamiltonian Parameters:
        scale: Threshold scale factor (multiplies epsilon)
        thresh_flag: True = ERF smooth function, False = hard step
        erf_sigma: Width of ERF smoothing function
    
    Experiment Control:
        repeat: Repeat index for statistical averaging
        seed: Random seed (auto-generated if None)
    
    Event Generation:
        n_events: Number of events per experiment
        n_particles_per_event: List of particle counts per event
        phi_max: Maximum phi angle for track generation
        theta_max: Maximum theta angle for track generation
    """
    
    # Physics parameters
    meas_error: float = 0.0
    coll_noise: float = 0.0
    ghost_rate: float = 0.0
    drop_rate: float = 0.0
    
    # Hamiltonian parameters
    scale: float = 1.0
    thresh_flag: bool = True
    erf_sigma: float = 1e-5
    
    # Experiment control
    repeat: int = 0
    seed: Optional[int] = None
    
    # Event generation
    n_events: int = 3
    n_particles_per_event: List[int] = field(default_factory=lambda: [5, 3, 5])
    phi_max: float = 0.02
    theta_max: float = 0.2
    
    # Hamiltonian tuning
    gamma: float = 2.0
    delta: float = 1.0
    theta_min: float = 0.000015
    
    # Detector (use default if not specified)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    
    def __post_init__(self):
        """Generate seed if not provided."""
        if self.seed is None:
            self.seed = self._generate_seed()
    
    def _generate_seed(self) -> int:
        """Generate deterministic seed from parameters for reproducibility."""
        s = f"{self.meas_error}|{self.coll_noise}|{self.ghost_rate}|{self.drop_rate}|{self.repeat}"
        return (abs(hash(s)) % (2**32 - 1)) or 123456789
    
    @property
    def total_particles(self) -> int:
        return sum(self.n_particles_per_event)
    
    @property
    def epsilon(self) -> float:
        """Compute the epsilon threshold from physics parameters."""
        theta_s = self.scale * self.coll_noise
        theta_r = np.arctan((self.scale * self.meas_error) / self.detector.dz_mm) \
                  if self.detector.dz_mm != 0 else 0.0
        theta_m = self.theta_min
        return np.sqrt(theta_s**2 + theta_r**2 + theta_m**2)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['epsilon'] = self.epsilon
        d['total_particles'] = self.total_particles
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentConfig':
        """Create from dictionary."""
        # Remove computed properties
        d = d.copy()
        d.pop('epsilon', None)
        d.pop('total_particles', None)
        
        # Handle nested detector config
        if 'detector' in d and isinstance(d['detector'], dict):
            d['detector'] = DetectorConfig(**d['detector'])
        
        return cls(**d)
    
    def get_filename_tag(self) -> str:
        """Generate a unique filename tag for this configuration."""
        return (
            f"m{self.meas_error}_c{self.coll_noise}_"
            f"g{self.ghost_rate}_d{self.drop_rate}_"
            f"r{self.repeat}_s{self.scale}_"
            f"t_{int(self.thresh_flag)}_e_{self.erf_sigma}_"
            f"seed{self.seed}"
        )


@dataclass
class ParameterGrid:
    """
    Define a grid of experiment configurations for batch runs.
    
    Example:
        grid = ParameterGrid(
            meas_errors=[0, 1e-5, 1e-4],
            coll_noises=[1e-5, 1e-4],
            scales=[1, 2, 3, 4, 5],
            repeats=10
        )
        configs = grid.generate_configs()
    """
    
    meas_errors: List[float] = field(default_factory=lambda: [0.0])
    coll_noises: List[float] = field(default_factory=lambda: [0.0])
    ghost_rates: List[float] = field(default_factory=lambda: [0.0])
    drop_rates: List[float] = field(default_factory=lambda: [0.0])
    scales: List[float] = field(default_factory=lambda: [1.0])
    thresh_flags: List[bool] = field(default_factory=lambda: [True])
    erf_sigmas: List[float] = field(default_factory=lambda: [1e-5])
    repeats: int = 1
    
    batch_size: int = 10
    
    def generate_configs(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations from the grid."""
        import itertools
        
        configs = []
        for meas, coll, ghost, drop, scale, thresh, erf_s, rep in itertools.product(
            self.meas_errors,
            self.coll_noises,
            self.ghost_rates,
            self.drop_rates,
            self.scales,
            self.thresh_flags,
            self.erf_sigmas,
            range(self.repeats)
        ):
            configs.append(ExperimentConfig(
                meas_error=meas,
                coll_noise=coll,
                ghost_rate=ghost,
                drop_rate=drop,
                scale=scale,
                thresh_flag=thresh,
                erf_sigma=erf_s,
                repeat=rep,
            ))
        
        return configs
    
    def generate_batches(self) -> List[List[ExperimentConfig]]:
        """Split configurations into batches for parallel execution."""
        configs = self.generate_configs()
        batches = []
        for i in range(0, len(configs), self.batch_size):
            batches.append(configs[i:i + self.batch_size])
        return batches
    
    def to_dataframe(self):
        """Convert to pandas DataFrame for CSV export."""
        import pandas as pd
        
        configs = self.generate_configs()
        rows = []
        for i, cfg in enumerate(configs):
            row = cfg.to_dict()
            row['row_id'] = i
            row['batch'] = i // self.batch_size
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save(self, path: str):
        """Save parameter grid to CSV file."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        
        # Also save batch list
        n_batches = (len(df) + self.batch_size - 1) // self.batch_size
        batch_file = path.replace('.csv', '_batches.txt')
        with open(batch_file, 'w') as f:
            for i in range(n_batches):
                f.write(f"{i}\n")
        
        return df
