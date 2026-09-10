from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


Vertex = Tuple[int, int]


@dataclass(frozen=True)
class SupportConfig:
    history_steps: int
    forecast_steps: int
    dt_hours: float
    max_lag_steps: int
    nearest_sources: int
    max_in_edges: int
    max_faces_per_target: int
    sigma_x: float


@dataclass(frozen=True)
class Edge:
    source: Vertex
    target: Vertex
    lag_steps: int
    residual: float
    score: float
    features: np.ndarray


@dataclass(frozen=True)
class Face:
    source: Vertex
    middle: Vertex
    target: Vertex
    first_edge: int
    second_edge: int
    direct_edge: int
    score: float
    features: np.ndarray


@dataclass
class TransportSupport:
    vertices: List[Vertex]
    edges: List[Edge]
    faces: List[Face]
    incoming_edges: Dict[Vertex, List[int]]
    incoming_faces: Dict[Vertex, List[int]]
