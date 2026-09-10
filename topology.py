from collections import defaultdict
from typing import List, Sequence, Tuple

import numpy as np

from .data import Edge, Face, SupportConfig, TransportSupport


def _time_index(t: int, cfg: SupportConfig) -> int:
    return t + cfg.history_steps


def _interval_velocity(
    wind: np.ndarray,
    i: int,
    j: int,
    r: int,
    cfg: SupportConfig,
) -> np.ndarray:
    a = _time_index(r, cfg)
    b = _time_index(r + 1, cfg)
    return 0.25 * (wind[i, a] + wind[i, b] + wind[j, a] + wind[j, b])


def _fit_continuous_delay(
    farm_xy: np.ndarray,
    wind: np.ndarray,
    i: int,
    j: int,
    q: int,
    cfg: SupportConfig,
) -> Tuple[float, float, int] | None:
    max_steps = min(cfg.max_lag_steps, q + cfg.history_steps - 1)
    if max_steps < 1:
        return None
    velocities = [
        _interval_velocity(wind, i, j, q - k - 1, cfg)
        for k in range(max_steps)
    ]
    if not all(np.isfinite(v).all() for v in velocities):
        return None
    displacement = farm_xy[j] - farm_xy[i]
    accumulated = np.zeros(2, dtype=float)
    best_tau = 0.0
    best_residual = float(np.linalg.norm(displacement))
    for k, velocity in enumerate(velocities):
        denominator = float(np.dot(velocity, velocity))
        if denominator > 0.0:
            theta = float(
                np.clip(
                    np.dot(displacement - accumulated, velocity) / denominator,
                    0.0,
                    cfg.dt_hours,
                )
            )
        else:
            theta = 0.0
        candidate = accumulated + theta * velocity
        residual = float(np.linalg.norm(displacement - candidate))
        tau = k * cfg.dt_hours + theta
        if residual < best_residual or (
            np.isclose(residual, best_residual) and tau < best_tau
        ):
            best_tau = tau
            best_residual = residual
        accumulated = accumulated + cfg.dt_hours * velocity
    upper = max_steps * cfg.dt_hours
    if best_tau < cfg.dt_hours or np.isclose(best_tau, upper):
        return None
    if best_residual > 2.0 * cfg.sigma_x:
        return None
    return best_tau, best_residual, max_steps


def _discrete_displacement(
    wind: np.ndarray,
    i: int,
    j: int,
    q: int,
    lag_steps: int,
    cfg: SupportConfig,
) -> np.ndarray:
    result = np.zeros(2, dtype=float)
    for r in range(q - lag_steps, q):
        result = result + cfg.dt_hours * _interval_velocity(wind, i, j, r, cfg)
    return result


def _pair_edges(
    farm_xy: np.ndarray,
    wind: np.ndarray,
    i: int,
    j: int,
    q: int,
    cfg: SupportConfig,
) -> List[Edge]:
    fitted = _fit_continuous_delay(farm_xy, wind, i, j, q, cfg)
    if fitted is None:
        return []
    tau_hat, _, max_steps = fitted
    displacement = farm_xy[j] - farm_xy[i]
    result = []
    for m in range(1, max_steps + 1):
        lag_hours = m * cfg.dt_hours
        if abs(lag_hours - tau_hat) > cfg.dt_hours:
            continue
        moved = _discrete_displacement(wind, i, j, q, m, cfg)
        residual = float(np.linalg.norm(displacement - moved))
        if residual > 2.0 * cfg.sigma_x:
            continue
        if float(np.dot(displacement, moved)) <= 0.0:
            continue
        score = float(
            np.exp(
                -(residual**2) / (2.0 * cfg.sigma_x**2)
                - ((lag_hours - tau_hat) ** 2) / (2.0 * cfg.dt_hours**2)
            )
        )
        p = q - m
        source_wind = wind[i, _time_index(p, cfg)]
        target_wind = wind[j, _time_index(q, cfg)]
        features = np.asarray(
            [
                displacement[0],
                displacement[1],
                lag_hours,
                residual,
                score,
                source_wind[0],
                source_wind[1],
                target_wind[0],
                target_wind[1],
            ],
            dtype=np.float32,
        )
        result.append(Edge((i, p), (j, q), m, residual, score, features))
    return result


def build_transport_support(
    farm_xy: np.ndarray,
    wind: np.ndarray,
    site_ids: Sequence[str | int],
    cfg: SupportConfig,
) -> TransportSupport:
    farm_xy = np.asarray(farm_xy, dtype=float)
    wind = np.asarray(wind, dtype=float)
    m_farms = farm_xy.shape[0]
    times = list(range(-cfg.history_steps + 1, cfg.forecast_steps + 1))
    nearest = {}
    for j in range(m_farms):
        choices = [i for i in range(m_farms) if i != j]
        choices.sort(
            key=lambda i: (
                float(np.linalg.norm(farm_xy[i] - farm_xy[j])),
                float(farm_xy[i, 0]),
                float(farm_xy[i, 1]),
                str(site_ids[i]),
            )
        )
        nearest[j] = choices[: cfg.nearest_sources]
    candidates = []
    for q in times:
        for j in range(m_farms):
            for i in nearest[j]:
                candidates.extend(_pair_edges(farm_xy, wind, i, j, q, cfg))
    grouped_edges = defaultdict(list)
    for edge in candidates:
        grouped_edges[edge.target].append(edge)
    edges = []
    for target, group in grouped_edges.items():
        group.sort(
            key=lambda edge: (
                -edge.score,
                float(farm_xy[edge.source[0], 0]),
                float(farm_xy[edge.source[0], 1]),
                str(site_ids[edge.source[0]]),
                edge.source[1],
            )
        )
        edges.extend(group[: cfg.max_in_edges])
    lookup = {(edge.source, edge.target): k for k, edge in enumerate(edges)}
    outgoing = defaultdict(list)
    for k, edge in enumerate(edges):
        outgoing[edge.source].append(k)
    grouped_faces = defaultdict(list)
    for first_index, first in enumerate(edges):
        a = first.source
        b = first.target
        for second_index in outgoing.get(b, []):
            second = edges[second_index]
            c = second.target
            if c[1] <= 0 or not a[1] < b[1] < c[1]:
                continue
            if len({a[0], b[0], c[0]}) != 3:
                continue
            direct_index = lookup.get((a, c))
            if direct_index is None:
                continue
            direct = edges[direct_index]
            score = float((first.score * second.score * direct.score) ** (1.0 / 3.0))
            features = np.asarray(
                [
                    first.lag_steps,
                    second.lag_steps,
                    direct.lag_steps,
                    first.residual,
                    second.residual,
                    direct.residual,
                    first.score,
                    second.score,
                    direct.score,
                    score,
                ],
                dtype=np.float32,
            )
            grouped_faces[c].append(
                Face(a, b, c, first_index, second_index, direct_index, score, features)
            )
    faces = []
    for target, group in grouped_faces.items():
        group.sort(
            key=lambda face: (
                -face.score,
                float(farm_xy[face.source[0], 0]),
                float(farm_xy[face.source[0], 1]),
                str(site_ids[face.source[0]]),
                float(farm_xy[face.middle[0], 0]),
                float(farm_xy[face.middle[0], 1]),
                str(site_ids[face.middle[0]]),
                face.source[1],
                face.middle[1],
            )
        )
        faces.extend(group[: cfg.max_faces_per_target])
    incoming_edges = defaultdict(list)
    incoming_faces = defaultdict(list)
    for k, edge in enumerate(edges):
        incoming_edges[edge.target].append(k)
    for k, face in enumerate(faces):
        incoming_faces[face.target].append(k)
    vertices = [(i, s) for s in times for i in range(m_farms)]
    return TransportSupport(
        vertices,
        edges,
        faces,
        dict(incoming_edges),
        dict(incoming_faces),
    )
