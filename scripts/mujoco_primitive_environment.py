"""Validate MotionBenchMaker primitives and add them to a MuJoCo model spec."""

from __future__ import annotations

import math
import re


ENVIRONMENT_KEYS = ("sphere", "cylinder", "box")


def _finite_vector(value, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must be an array of {length} numbers")
    result = [float(component) for component in value]
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def normalize_primitive_environment(
    raw_environment,
    label: str,
) -> dict[str, list]:
    """Return a validated, numeric sphere/cylinder/box environment."""
    if not isinstance(raw_environment, dict):
        raise ValueError(f"{label} is missing its primitive environment")

    environment: dict[str, list] = {key: [] for key in ENVIRONMENT_KEYS}
    for key in ENVIRONMENT_KEYS:
        primitives = raw_environment.get(key, [])
        if not isinstance(primitives, list):
            raise ValueError(f"{label} environment.{key} must be an array")
        for index, primitive in enumerate(primitives):
            primitive_label = f"{label} environment.{key}[{index}]"
            if not isinstance(primitive, dict):
                raise ValueError(f"{primitive_label} must be a JSON object")
            normalized = {
                "name": str(primitive.get("name", f"{key}_{index}")),
                "position": _finite_vector(
                    primitive.get("position"), 3, f"{primitive_label}.position"
                ),
            }
            if key == "sphere":
                radius = float(primitive.get("radius", math.nan))
                if not math.isfinite(radius) or radius <= 0.0:
                    raise ValueError(f"{primitive_label}.radius must be positive")
                normalized["radius"] = radius
            elif key == "cylinder":
                normalized["orientation_euler_xyz"] = _finite_vector(
                    primitive.get("orientation_euler_xyz"),
                    3,
                    f"{primitive_label}.orientation_euler_xyz",
                )
                radius = float(primitive.get("radius", math.nan))
                length = float(primitive.get("length", math.nan))
                if not math.isfinite(radius) or radius <= 0.0:
                    raise ValueError(f"{primitive_label}.radius must be positive")
                if not math.isfinite(length) or length <= 0.0:
                    raise ValueError(f"{primitive_label}.length must be positive")
                normalized["radius"] = radius
                normalized["length"] = length
            else:
                normalized["orientation_euler_xyz"] = _finite_vector(
                    primitive.get("orientation_euler_xyz"),
                    3,
                    f"{primitive_label}.orientation_euler_xyz",
                )
                half_extents = _finite_vector(
                    primitive.get("half_extents"),
                    3,
                    f"{primitive_label}.half_extents",
                )
                if any(extent <= 0.0 for extent in half_extents):
                    raise ValueError(
                        f"{primitive_label}.half_extents must be positive"
                    )
                normalized["half_extents"] = half_extents
            environment[key].append(normalized)
    return environment


def empty_primitive_environment() -> dict[str, list]:
    return {key: [] for key in ENVIRONMENT_KEYS}


def primitive_count(environment: dict[str, list]) -> int:
    return sum(len(environment[key]) for key in ENVIRONMENT_KEYS)


def _geom_name(kind: str, index: int, primitive: dict) -> str:
    source_name = re.sub(r"[^A-Za-z0-9_]+", "_", primitive["name"]).strip("_")
    if not source_name:
        source_name = kind
    return f"planning_{kind}_{index}_{source_name}"[:120]


def add_primitive_environment(
    mujoco,
    spec,
    environment: dict[str, list],
    *,
    physical: bool,
) -> None:
    """Add validated primitives as static world geoms to ``spec``."""
    conaffinity = 1 if physical else 0
    for index, sphere in enumerate(environment["sphere"]):
        spec.worldbody.add_geom(
            name=_geom_name("sphere", index, sphere),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=sphere["position"],
            size=[sphere["radius"], 0.0, 0.0],
            contype=0,
            conaffinity=conaffinity,
            density=0.0,
            rgba=[0.85, 0.25, 0.20, 0.65],
        )

    for index, cylinder in enumerate(environment["cylinder"]):
        spec.worldbody.add_geom(
            name=_geom_name("cylinder", index, cylinder),
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=cylinder["position"],
            euler=cylinder["orientation_euler_xyz"],
            size=[cylinder["radius"], 0.5 * cylinder["length"], 0.0],
            contype=0,
            conaffinity=conaffinity,
            density=0.0,
            rgba=[0.85, 0.45, 0.15, 0.65],
        )

    for index, box in enumerate(environment["box"]):
        normalized_name = box["name"].lower()
        is_obstacle = normalized_name.startswith(("object", "obstacle"))
        spec.worldbody.add_geom(
            name=_geom_name("box", index, box),
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=box["position"],
            euler=box["orientation_euler_xyz"],
            size=box["half_extents"],
            contype=0,
            conaffinity=conaffinity,
            density=0.0,
            rgba=(
                [0.85, 0.20, 0.15, 0.65]
                if is_obstacle
                else [0.45, 0.32, 0.18, 0.65]
            ),
        )
