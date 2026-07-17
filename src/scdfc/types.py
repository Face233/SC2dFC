from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubjectRunSample:
    subject_id: str
    run: str
    sc: np.ndarray
    fc_warmup: np.ndarray
    fc_future: np.ndarray
    window_starts: np.ndarray


@dataclass(frozen=True)
class Prediction:
    fc_z_edges: object
    fc_matrices: object
    latent: object | None = None
