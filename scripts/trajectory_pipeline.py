#!/usr/bin/env python3
"""Quintic-Hermite geometry and TOPP-RA timing for visualization paths."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


QUINTIC_HERMITE_FORMAT = "quintic_hermite_v1"
DEFAULT_VALIDATION_SAMPLES_PER_SEGMENT = 4
TOPPRA_LIMIT_SAFETY_FACTORS = (
    0.98, 0.95, 0.92, 0.88, 0.82, 0.70, 0.50, 0.35
)
TOPPRA_GRIDPOINTS_PER_SPLINE_SEGMENT = 16


class GeometricPathWaypoints(list):
    """Waypoint list carrying an optional prevalidated geometric path."""

    def __init__(
        self,
        values,
        geometric_path: dict[str, Any] | None = None,
        path_smoothing: bool = True,
    ) -> None:
        super().__init__(values)
        self.geometric_path = geometric_path
        self.path_smoothing = bool(path_smoothing)


def _require_scipy_bpoly():
    try:
        from scipy.interpolate import BPoly
    except ImportError as error:
        raise RuntimeError(
            "quintic Hermite interpolation requires SciPy; install "
            "requirements-visualization.txt"
        ) from error
    return BPoly


def _require_toppra():
    try:
        import toppra
        import toppra.algorithm as algorithm
        import toppra.constraint as constraint
        from toppra.interpolator import AbstractGeometricPath
    except ImportError as error:
        raise RuntimeError(
            "TOPP-RA time parameterization requires toppra; install "
            "requirements-visualization.txt"
        ) from error
    return toppra, algorithm, constraint, AbstractGeometricPath


def _finite_coordinate_limits(
    value: float | list[float] | np.ndarray,
    dimension: int,
    description: str,
) -> np.ndarray:
    limits = np.asarray(value, dtype=np.float64)
    if limits.ndim == 0:
        limits = np.full(dimension, float(limits), dtype=np.float64)
    if limits.shape != (dimension,):
        raise ValueError(
            f"{description} must be a scalar or contain {dimension} values"
        )
    if not np.isfinite(limits).all() or np.any(limits <= 0.0):
        raise ValueError(f"{description} values must be finite and positive")
    return limits


def _deduplicate_waypoints(waypoints: np.ndarray) -> np.ndarray:
    if len(waypoints) < 2:
        return waypoints
    keep = np.concatenate(
        (
            np.asarray([True]),
            np.linalg.norm(np.diff(waypoints, axis=0), axis=1) > 1.0e-12,
        )
    )
    return waypoints[keep]


def _chord_length_knots(positions: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if np.any(lengths <= 0.0) or not np.isfinite(lengths).all():
        raise ValueError("quintic Hermite waypoints must be distinct and finite")
    return np.concatenate((np.asarray([0.0]), np.cumsum(lengths)))


def _estimate_knot_derivatives(
    positions: np.ndarray,
    knots: np.ndarray,
    derivative_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    edge_order = 2 if len(positions) >= 3 else 1
    first = np.gradient(positions, knots, axis=0, edge_order=edge_order)
    second = np.gradient(first, knots, axis=0, edge_order=edge_order)
    return derivative_scale * first, derivative_scale * second


def _make_bpoly(
    knots: np.ndarray,
    positions: np.ndarray,
    first_derivatives: np.ndarray,
    second_derivatives: np.ndarray,
):
    BPoly = _require_scipy_bpoly()
    derivative_data = [
        [positions[index], first_derivatives[index], second_derivatives[index]]
        for index in range(len(knots))
    ]
    # Three endpoint conditions at each side produce degree five on every
    # interval. BPoly uses the Bernstein polynomial basis; it is not BSpline.
    return BPoly.from_derivatives(
        knots,
        derivative_data,
        orders=5,
        extrapolate=False,
    )


def build_quintic_hermite_geometry(
    waypoints: list[list[float]] | np.ndarray,
    derivative_scale: float = 1.0,
    validation_samples_per_segment: int = (
        DEFAULT_VALIDATION_SAMPLES_PER_SEGMENT
    ),
) -> tuple[dict[str, Any], np.ndarray]:
    """Build a C2 piecewise-quintic Hermite path and validation samples."""
    positions = np.asarray(waypoints, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] < 1:
        raise ValueError("quintic Hermite interpolation needs at least two points")
    if not np.isfinite(positions).all():
        raise ValueError("quintic Hermite waypoints must be finite")
    if not math.isfinite(derivative_scale) or not 0.0 <= derivative_scale <= 1.0:
        raise ValueError("derivative scale must be finite and in [0, 1]")
    if validation_samples_per_segment < 1:
        raise ValueError("validation samples per segment must be positive")

    positions = _deduplicate_waypoints(positions)
    if len(positions) < 2:
        raise ValueError("quintic Hermite path has no nonzero segment")
    knots = _chord_length_knots(positions)
    first, second = _estimate_knot_derivatives(
        positions,
        knots,
        derivative_scale,
    )
    polynomial = _make_bpoly(knots, positions, first, second)

    sample_positions: list[float] = []
    for segment in range(len(knots) - 1):
        sample_positions.extend(
            np.linspace(
                knots[segment],
                knots[segment + 1],
                validation_samples_per_segment,
                endpoint=False,
            ).tolist()
        )
    sample_positions.append(float(knots[-1]))
    validation_samples = np.asarray(
        polynomial(np.asarray(sample_positions)),
        dtype=np.float64,
    )
    validation_samples[0] = positions[0]
    validation_samples[-1] = positions[-1]

    geometry: dict[str, Any] = {
        "format": QUINTIC_HERMITE_FORMAT,
        "kind": "quintic_hermite",
        "basis": "bernstein_piecewise_polynomial",
        "degree": 5,
        "continuity": "C2",
        "derivative_scale": float(derivative_scale),
        "validation_samples_per_segment": int(
            validation_samples_per_segment
        ),
        "knots": knots.tolist(),
        "positions": positions.tolist(),
        "first_derivatives": first.tolist(),
        "second_derivatives": second.tolist(),
    }
    return geometry, validation_samples


def quintic_hermite_polynomial(geometry: dict[str, Any]):
    if geometry.get("format") != QUINTIC_HERMITE_FORMAT:
        raise ValueError("unsupported geometric path format")
    if geometry.get("kind") != "quintic_hermite" or geometry.get("degree") != 5:
        raise ValueError("geometric path is not a quintic Hermite spline")
    if geometry.get("basis") != "bernstein_piecewise_polynomial":
        raise ValueError(
            "quintic Hermite geometry must use the Bernstein polynomial basis; "
            "B-spline geometry is not accepted"
        )
    knots = np.asarray(geometry["knots"], dtype=np.float64)
    positions = np.asarray(geometry["positions"], dtype=np.float64)
    first = np.asarray(geometry["first_derivatives"], dtype=np.float64)
    second = np.asarray(geometry["second_derivatives"], dtype=np.float64)
    if knots.ndim != 1 or positions.ndim != 2 or len(knots) != len(positions):
        raise ValueError("invalid quintic Hermite geometry dimensions")
    if first.shape != positions.shape or second.shape != positions.shape:
        raise ValueError("invalid quintic Hermite derivative dimensions")
    if not (
        np.isfinite(knots).all()
        and np.isfinite(positions).all()
        and np.isfinite(first).all()
        and np.isfinite(second).all()
    ):
        raise ValueError("quintic Hermite geometry must be finite")
    if np.any(np.diff(knots) <= 0.0):
        raise ValueError("quintic Hermite knots must be strictly increasing")
    return _make_bpoly(knots, positions, first, second)


def _quintic_path_type():
    _, _, _, AbstractGeometricPath = _require_toppra()

    class QuinticHermitePath(AbstractGeometricPath):
        def __init__(self, geometry: dict[str, Any]) -> None:
            self._geometry = geometry
            self._polynomial = quintic_hermite_polynomial(geometry)
            self._knots = np.asarray(geometry["knots"], dtype=np.float64)
            self._positions = np.asarray(
                geometry["positions"], dtype=np.float64
            )

        def __call__(self, path_positions, order: int = 0) -> np.ndarray:
            if order not in (0, 1, 2):
                raise ValueError("quintic Hermite path supports orders 0, 1, 2")
            bounded_positions = np.clip(
                path_positions,
                self._knots[0],
                self._knots[-1],
            )
            values = np.asarray(
                self._polynomial(bounded_positions, nu=order),
                dtype=np.float64,
            )
            if not np.isfinite(values).all():
                raise ValueError("path evaluation is outside its knot interval")
            return values

        @property
        def dof(self) -> int:
            return int(self._positions.shape[1])

        @property
        def path_interval(self) -> np.ndarray:
            return np.asarray([self._knots[0], self._knots[-1]])

        @property
        def waypoints(self):
            return self._knots.copy(), self._positions.copy()

    return QuinticHermitePath


@dataclass(frozen=True)
class ToppraTrajectory:
    trajectory: Any
    geometric_path: Any
    source_waypoints: np.ndarray
    path_parameter_scale: float
    peak_velocity: float
    peak_acceleration: float
    velocity_limit_utilization: float
    acceleration_limit_utilization: float

    @property
    def duration(self) -> float:
        return float(self.trajectory.path_interval[1])


def time_parameterize_quintic_path(
    geometry: dict[str, Any],
    maximum_velocity: float | list[float] | np.ndarray,
    maximum_acceleration: float | list[float] | np.ndarray,
    fps: float | None = None,
    require_cuda_revalidation: bool = True,
) -> ToppraTrajectory:
    """Run TOPP-RA on a previously validated quintic Hermite path."""
    _, algorithm, constraint, _ = _require_toppra()
    if require_cuda_revalidation and not geometry.get("cuda_revalidated", False):
        raise ValueError(
            "TOPP-RA requires a quintic spline marked as CUDA revalidated"
        )
    PathType = _quintic_path_type()
    derivative_scale = float(geometry.get("derivative_scale", 1.0))
    # Small Hermite tangents keep a tight spline inside narrow valid regions,
    # but make the original path coordinate numerically singular for TOPP-RA.
    # Affinely scaling knots and transforming their derivatives preserves the
    # exact geometric curve while giving TOPP-RA a better-conditioned path
    # coordinate.
    path_parameter_scale = max(
        1.0e-4,
        min(1.0, math.sqrt(max(0.0, derivative_scale))),
    )
    parameterized_geometry = dict(geometry)
    parameterized_geometry["knots"] = (
        np.asarray(geometry["knots"], dtype=np.float64) *
        path_parameter_scale
    ).tolist()
    parameterized_geometry["first_derivatives"] = (
        np.asarray(geometry["first_derivatives"], dtype=np.float64) /
        path_parameter_scale
    ).tolist()
    parameterized_geometry["second_derivatives"] = (
        np.asarray(geometry["second_derivatives"], dtype=np.float64) /
        (path_parameter_scale * path_parameter_scale)
    ).tolist()
    path = PathType(parameterized_geometry)
    velocity_limits = _finite_coordinate_limits(
        maximum_velocity,
        path.dof,
        "trajectory velocity limits",
    )
    acceleration_limits = _finite_coordinate_limits(
        maximum_acceleration,
        path.dof,
        "trajectory acceleration limits",
    )
    if fps is not None and (not math.isfinite(fps) or fps <= 0.0):
        raise ValueError("trajectory fps must be positive")

    knots = np.asarray(parameterized_geometry["knots"], dtype=np.float64)
    gridpoint_parts = [
        np.linspace(
            knots[index],
            knots[index + 1],
            TOPPRA_GRIDPOINTS_PER_SPLINE_SEGMENT,
            endpoint=False,
        )
        for index in range(len(knots) - 1)
    ]
    gridpoints = np.concatenate((*gridpoint_parts, knots[-1:]))
    trajectory = None
    velocities = None
    accelerations = None
    velocity_utilization = math.inf
    acceleration_utilization = math.inf
    last_return_code: Any = "not-run"
    for safety_factor in TOPPRA_LIMIT_SAFETY_FACTORS:
        velocity_constraint = constraint.JointVelocityConstraint(
            safety_factor *
            np.column_stack((-velocity_limits, velocity_limits))
        )
        acceleration_constraint = constraint.JointAccelerationConstraint(
            safety_factor *
            np.column_stack((-acceleration_limits, acceleration_limits))
        )
        instance = algorithm.TOPPRA(
            [velocity_constraint, acceleration_constraint],
            path,
            gridpoints=gridpoints,
            solver_wrapper="seidel",
            parametrizer="ParametrizeConstAccel",
        )
        candidate = instance.compute_trajectory(0.0, 0.0)
        last_return_code = getattr(
            instance.problem_data,
            "return_code",
            "unknown",
        )
        if candidate is None:
            continue
        duration = float(candidate.path_interval[1])
        if not math.isfinite(duration) or duration <= 0.0:
            continue
        diagnostic_count = max(
            2000,
            4 * len(gridpoints),
            int(math.ceil(
                duration * (fps if fps is not None else 60.0) * 2.0
            )),
        )
        diagnostic_times = np.linspace(0.0, duration, diagnostic_count)
        candidate_velocities = np.asarray(candidate.evald(diagnostic_times))
        candidate_accelerations = np.asarray(candidate.evaldd(diagnostic_times))
        candidate_velocity_utilization = float(
            np.max(
                np.abs(candidate_velocities) / velocity_limits[None, :]
            )
        )
        candidate_acceleration_utilization = float(
            np.max(
                np.abs(candidate_accelerations) /
                acceleration_limits[None, :]
            )
        )
        velocity_utilization = candidate_velocity_utilization
        acceleration_utilization = candidate_acceleration_utilization
        if candidate_velocity_utilization <= 1.0 + 1.0e-3 and (
            candidate_acceleration_utilization <= 1.0 + 1.0e-3
        ):
            trajectory = candidate
            velocities = candidate_velocities
            accelerations = candidate_accelerations
            break
    if trajectory is None or velocities is None or accelerations is None:
        raise RuntimeError(
            "TOPP-RA failed to produce a limit-valid trajectory "
            f"(return_code={last_return_code}, "
            f"velocity_utilization={velocity_utilization:.6g}, "
            f"acceleration_utilization={acceleration_utilization:.6g})"
        )
    positions = np.asarray(geometry["positions"], dtype=np.float64)
    return ToppraTrajectory(
        trajectory=trajectory,
        geometric_path=path,
        source_waypoints=positions,
        path_parameter_scale=path_parameter_scale,
        peak_velocity=float(np.max(np.abs(velocities))),
        peak_acceleration=float(np.max(np.abs(accelerations))),
        velocity_limit_utilization=velocity_utilization,
        acceleration_limit_utilization=acceleration_utilization,
    )


def sample_toppra_trajectory(
    trajectory: ToppraTrajectory,
    sample_time: float,
) -> list[float]:
    time_value = min(max(float(sample_time), 0.0), trajectory.duration)
    return np.asarray(trajectory.trajectory.eval(time_value)).tolist()


def toppra_trajectory_frames(trajectory: ToppraTrajectory, fps: float):
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("trajectory fps must be positive")
    frame_count = max(1, int(math.ceil(trajectory.duration * fps)))
    for frame_index in range(1, frame_count + 1):
        yield sample_toppra_trajectory(
            trajectory,
            min(frame_index / fps, trajectory.duration),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--derivative-scale", required=True, type=float)
    parser.add_argument(
        "--validation-samples-per-segment",
        type=int,
        default=DEFAULT_VALIDATION_SAMPLES_PER_SEGMENT,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    geometry, validation_samples = build_quintic_hermite_geometry(
        document["waypoints"],
        args.derivative_scale,
        args.validation_samples_per_segment,
    )
    output = {
        "geometric_path": geometry,
        "validation_samples": validation_samples.tolist(),
    }
    args.output.write_text(json.dumps(output), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
