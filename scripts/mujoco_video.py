"""Shared command-line and FFmpeg helpers for MuJoCo MP4 rendering."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from trajectory_pipeline import (
    GeometricPathWaypoints,
    ToppraTrajectory,
    build_quintic_hermite_geometry,
    sample_toppra_trajectory,
    time_parameterize_quintic_path,
    toppra_trajectory_frames,
)


DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_VIDEO_HEIGHT = 1080
DEFAULT_VIDEO_CRF = 12
DEFAULT_VIDEO_PRESET = "slow"
DEFAULT_VIDEO_SAMPLES = 8
DEFAULT_TRAJECTORY_PLAYBACK_RATE = 2.0
DEFAULT_TRAJECTORY_SPEED = 0.7 * DEFAULT_TRAJECTORY_PLAYBACK_RATE
DEFAULT_TRAJECTORY_ACCELERATION = (
    1.4 * DEFAULT_TRAJECTORY_PLAYBACK_RATE**2
)
VIDEO_VIEW_NAMES = ("front", "front_left", "front_right", "overhead")
VIDEO_VIEW_AZIMUTH_OFFSETS = {
    "front": 0.0,
    "front_left": 45.0,
    "front_right": 315.0,
    "overhead": 0.0,
    "left": 90.0,
    "rear": 180.0,
    "right": 270.0,
}
VIDEO_VIEW_ELEVATION_OFFSETS = {
    "overhead": -45.0,
}


@dataclass(frozen=True)
class LinearWaypointTrajectory:
    """Raw waypoint polyline with constant velocity on each segment."""

    waypoints: np.ndarray
    segment_durations: np.ndarray
    maximum_velocity: np.ndarray
    peak_velocity: float
    velocity_limit_utilization: float
    peak_acceleration: float = math.nan
    acceleration_limit_utilization: float = math.nan
    uses_toppra: bool = False

    @property
    def duration(self) -> float:
        return float(np.sum(self.segment_durations))


TimeParameterizedTrajectory = ToppraTrajectory | LinearWaypointTrajectory


def _finite_velocity_limits(
    value: float | list[float] | np.ndarray,
    dimension: int,
) -> np.ndarray:
    limits = np.asarray(value, dtype=np.float64)
    if limits.ndim == 0:
        limits = np.full(dimension, float(limits), dtype=np.float64)
    if limits.shape != (dimension,):
        raise ValueError(
            f"maximum velocity must be a scalar or contain {dimension} values"
        )
    if not np.isfinite(limits).all() or np.any(limits <= 0.0):
        raise ValueError("maximum velocity values must be finite and positive")
    return limits


def _linear_waypoint_trajectory(
    positions: np.ndarray,
    maximum_velocity: float | list[float] | np.ndarray,
    fps: float | None,
) -> LinearWaypointTrajectory:
    """Assign constant-speed durations without smoothing or TOPP-RA."""
    limits = _finite_velocity_limits(maximum_velocity, positions.shape[1])
    changes = np.abs(np.diff(positions, axis=0))
    raw_durations = np.max(changes / limits[np.newaxis, :], axis=1)
    minimum_duration = 1.0 / fps if fps is not None else 1.0e-9
    if fps is not None:
        frame_counts = np.maximum(1, np.ceil(raw_durations * fps)).astype(int)
        durations = frame_counts.astype(np.float64) / fps
    else:
        durations = np.maximum(raw_durations, minimum_duration)
    segment_velocities = changes / durations[:, np.newaxis]
    peak_velocity = float(np.max(segment_velocities))
    utilization = float(
        np.max(segment_velocities / limits[np.newaxis, :])
    )
    return LinearWaypointTrajectory(
        waypoints=positions.copy(),
        segment_durations=durations,
        maximum_velocity=limits,
        peak_velocity=peak_velocity,
        velocity_limit_utilization=utilization,
    )


def time_parameterize_waypoints(
    waypoints: list[list[float]],
    maximum_velocity: float | list[float] | np.ndarray,
    maximum_acceleration: float | list[float] | np.ndarray = (
        DEFAULT_TRAJECTORY_ACCELERATION
    ),
    fps: float | None = None,
) -> TimeParameterizedTrajectory:
    """Prepare smoothed TOPP-RA or raw linear waypoint playback."""
    positions = np.asarray(waypoints, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] < 1:
        raise ValueError(
            "time parameterization requires at least two non-empty waypoints"
        )
    if not np.isfinite(positions).all():
        raise ValueError("time parameterization waypoints must be finite")
    if fps is not None and (not math.isfinite(fps) or fps <= 0.0):
        raise ValueError("trajectory fps must be positive")

    if not getattr(waypoints, "path_smoothing", True):
        return _linear_waypoint_trajectory(positions, maximum_velocity, fps)

    geometric_path = getattr(waypoints, "geometric_path", None)
    require_cuda_revalidation = geometric_path is not None
    if geometric_path is None:
        geometric_path, _ = build_quintic_hermite_geometry(
            positions,
            derivative_scale=1.0,
        )
    return time_parameterize_quintic_path(
        geometric_path,
        maximum_velocity,
        maximum_acceleration,
        fps,
        require_cuda_revalidation=require_cuda_revalidation,
    )


def sample_time_parameterized_trajectory(
    trajectory: TimeParameterizedTrajectory,
    sample_time: float,
) -> list[float]:
    if isinstance(trajectory, LinearWaypointTrajectory):
        time_value = min(max(float(sample_time), 0.0), trajectory.duration)
        elapsed = 0.0
        for segment_index, duration in enumerate(
            trajectory.segment_durations
        ):
            segment_end = elapsed + float(duration)
            if time_value <= segment_end or segment_index == len(
                trajectory.segment_durations
            ) - 1:
                fraction = min(
                    max((time_value - elapsed) / float(duration), 0.0),
                    1.0,
                )
                start = trajectory.waypoints[segment_index]
                end = trajectory.waypoints[segment_index + 1]
                return ((1.0 - fraction) * start + fraction * end).tolist()
            elapsed = segment_end
        return trajectory.waypoints[-1].tolist()
    return sample_toppra_trajectory(trajectory, sample_time)


def time_parameterized_frames(
    trajectory: TimeParameterizedTrajectory,
    fps: float,
):
    """Yield frame-rate samples of a prepared trajectory."""
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("trajectory fps must be positive")
    if isinstance(trajectory, LinearWaypointTrajectory):
        for segment_index, duration in enumerate(
            trajectory.segment_durations
        ):
            frame_count = max(1, int(math.ceil(float(duration) * fps - 1.0e-12)))
            start = trajectory.waypoints[segment_index]
            end = trajectory.waypoints[segment_index + 1]
            for frame_index in range(frame_count):
                fraction = frame_index / frame_count
                yield ((1.0 - fraction) * start + fraction * end).tolist()
        yield trajectory.waypoints[-1].tolist()
        return
    yield from toppra_trajectory_frames(trajectory, fps)


def continuous_trajectory_frames(
    waypoints: list[list[float]],
    fps: float,
    maximum_velocity: float | list[float] | np.ndarray,
    maximum_acceleration: float | list[float] | np.ndarray = (
        DEFAULT_TRAJECTORY_ACCELERATION
    ),
):
    """Time-parameterize a path and yield its frame-aligned configurations."""
    trajectory = time_parameterize_waypoints(
        waypoints,
        maximum_velocity,
        maximum_acceleration,
        fps,
    )
    yield from time_parameterized_frames(trajectory, fps)


def add_video_view_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--video-views",
        default=os.environ.get("PRRTC_VIDEO_VIEWS", "front"),
        help=(
            "Comma-separated camera views: front, front_left, front_right, "
            "overhead, left, rear, right, or all (default: front). "
            "PRRTC_VIDEO_VIEWS provides the same option."
        ),
    )


def add_video_arguments(
    parser: argparse.ArgumentParser,
    robot_environment_variable: str,
) -> None:
    """Add the common offscreen-video options to a visualizer parser."""
    video_default = os.environ.get(robot_environment_variable) or os.environ.get(
        "PRRTC_VIDEO"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(video_default) if video_default else None,
        help=(
            "Render one start-to-goal replay directly to an MP4 instead of "
            f"opening a viewer. {robot_environment_variable} or PRRTC_VIDEO "
            "provides the same option."
        ),
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=os.environ.get("PRRTC_VIDEO_WIDTH", str(DEFAULT_VIDEO_WIDTH)),
        help=(
            f"Recorded video width in pixels (default: {DEFAULT_VIDEO_WIDTH}; "
            "PRRTC_VIDEO_WIDTH provides the same option)."
        ),
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=os.environ.get("PRRTC_VIDEO_HEIGHT", str(DEFAULT_VIDEO_HEIGHT)),
        help=(
            f"Recorded video height in pixels (default: {DEFAULT_VIDEO_HEIGHT}; "
            "PRRTC_VIDEO_HEIGHT provides the same option)."
        ),
    )
    add_video_view_argument(parser)


def validate_video_views(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    raw_views = args.video_views
    if not isinstance(raw_views, str):
        parser.error("--video-views must be a comma-separated string")
    requested = [
        value.strip().lower()
        for value in raw_views.split(",")
        if value.strip()
    ]
    if not requested:
        parser.error("--video-views must select at least one camera view")
    aliases = {"back": "rear", "top": "overhead"}
    requested = [aliases.get(value, value) for value in requested]
    if "all" in requested:
        if len(requested) != 1:
            parser.error("--video-views all cannot be combined with other views")
        args.video_views = VIDEO_VIEW_NAMES
        return
    unknown = [
        value for value in requested if value not in VIDEO_VIEW_AZIMUTH_OFFSETS
    ]
    if unknown:
        parser.error(
            "--video-views contains unsupported views: " + ", ".join(unknown)
        )
    args.video_views = tuple(dict.fromkeys(requested))


def validate_video_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """Validate options shared by every offscreen MP4 renderer."""
    if args.video is not None and args.validate_only:
        parser.error("--validate-only and --video cannot be used together")
    if args.video is not None and args.video.suffix.lower() != ".mp4":
        parser.error("--video output must use the .mp4 extension")
    if (
        args.video_width <= 0
        or args.video_height <= 0
        or args.video_width % 2 != 0
        or args.video_height % 2 != 0
    ):
        parser.error(
            "--video-width and --video-height must be positive even integers"
        )
    validate_video_views(parser, args)


def video_view_azimuth(base_azimuth: float, view: str) -> float:
    return (base_azimuth + VIDEO_VIEW_AZIMUTH_OFFSETS[view]) % 360.0


def video_view_elevation(base_elevation: float, view: str) -> float:
    return base_elevation + VIDEO_VIEW_ELEVATION_OFFSETS.get(view, 0.0)


def video_output_paths(
    output_path: Path,
    views: tuple[str, ...],
) -> dict[str, Path]:
    resolved = output_path.expanduser().resolve()
    if len(views) == 1:
        return {views[0]: resolved}
    return {
        view: resolved.with_name(f"{resolved.stem}_{view}{resolved.suffix}")
        for view in views
    }


def configure_camera(
    mujoco,
    camera,
    lookat: tuple[float, float, float],
    distance: float,
    azimuth: float,
    elevation: float,
) -> None:
    """Configure a deterministic free camera for viewer and video parity."""
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation


def configure_model_render_quality(model) -> int:
    """Apply the requested offscreen multisample count and return it."""
    raw_samples = os.environ.get(
        "PRRTC_VIDEO_SAMPLES", str(DEFAULT_VIDEO_SAMPLES)
    )
    try:
        samples = int(raw_samples)
    except ValueError as error:
        raise ValueError("PRRTC_VIDEO_SAMPLES must be an integer") from error
    if samples not in (0, 2, 4, 8, 16):
        raise ValueError(
            "PRRTC_VIDEO_SAMPLES must be one of: 0, 2, 4, 8, 16"
        )
    model.vis.quality.offsamples = samples
    return samples


@lru_cache(maxsize=8)
def _video_overlay_font(font_size: int):
    try:
        from PIL import ImageFont
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required to draw text into recorded videos; "
            "install the visualization requirements and retry"
        ) from error

    for font_name in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_video_text_overlay(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a readable, resolution-scaled label into a frame's top right."""
    if not text:
        return frame
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("video text overlays require an uint8 RGB frame")

    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required to draw text into recorded videos; "
            "install the visualization requirements and retry"
        ) from error

    height, width = frame.shape[:2]
    font = _video_overlay_font(max(16, round(height * 0.030)))
    margin = max(10, round(height * 0.020))
    padding_x = max(10, round(height * 0.012))
    padding_y = max(7, round(height * 0.008))

    canvas = Image.fromarray(frame).convert("RGBA")
    panel = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    text_box = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        spacing=max(2, round(height * 0.004)),
        align="right",
    )
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    panel_right = width - margin
    panel_left = max(margin, panel_right - text_width - 2 * padding_x)
    panel_top = margin
    panel_bottom = panel_top + text_height + 2 * padding_y
    radius = max(5, round(height * 0.007))
    draw.rounded_rectangle(
        (panel_left, panel_top, panel_right, panel_bottom),
        radius=radius,
        fill=(0, 0, 0, 172),
        outline=(255, 255, 255, 80),
        width=max(1, round(height / 720)),
    )
    draw.multiline_text(
        (panel_right - padding_x, panel_top + padding_y - text_box[1]),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        anchor="ra",
        spacing=max(2, round(height * 0.004)),
        align="right",
        stroke_width=max(1, round(height / 1080)),
        stroke_fill=(0, 0, 0, 220),
    )
    return np.asarray(Image.alpha_composite(canvas, panel).convert("RGB"))


class FfmpegVideoWriter:
    """Stream RGB frames to an H.264 MP4 without buffering them in memory."""

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "FFmpeg is required for MP4 output; install ffmpeg and retry"
            )
        self.output_path = output_path.expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.frame_count = 0
        raw_crf = os.environ.get("PRRTC_VIDEO_CRF", str(DEFAULT_VIDEO_CRF))
        try:
            crf = int(raw_crf)
        except ValueError as error:
            raise ValueError("PRRTC_VIDEO_CRF must be an integer") from error
        if crf < 0 or crf > 51:
            raise ValueError("PRRTC_VIDEO_CRF must be between 0 and 51")
        preset = os.environ.get("PRRTC_VIDEO_PRESET", DEFAULT_VIDEO_PRESET)
        supported_presets = {
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow",
        }
        if preset not in supported_presets:
            raise ValueError(
                "PRRTC_VIDEO_PRESET must be a valid x264 preset"
            )
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                f"{fps:.12g}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-tune",
                "animation",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.output_path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed video encoder")
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"video frame has shape {frame.shape}, expected "
                f"({self.height}, {self.width}, 3)"
            )
        if frame.dtype != np.uint8:
            raise ValueError("video frames must use uint8 RGB pixels")
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as error:
            raise RuntimeError("FFmpeg stopped while encoding the MP4") from error
        self.frame_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        assert self._process.stdin is not None
        assert self._process.stderr is not None
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        stderr = self._process.stderr.read().decode("utf-8", errors="replace")
        return_code = self._process.wait()
        if return_code != 0:
            detail = stderr.strip() or f"exit status {return_code}"
            raise RuntimeError(f"FFmpeg failed to encode MP4: {detail}")

    def __enter__(self):
        return self

    def __exit__(self, exception_type, _exception, _traceback) -> bool:
        try:
            self.close()
        except Exception:
            if exception_type is None:
                raise
        return False


class MultiViewVideoWriter:
    """Write synchronized frames from one simulation to one MP4 per view."""

    def __init__(
        self,
        output_path: Path,
        views: tuple[str, ...],
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.output_paths = video_output_paths(output_path, views)
        self.width = width
        self.height = height
        self.fps = fps
        self._writers: dict[str, FfmpegVideoWriter] = {}

    @property
    def frame_count(self) -> int:
        counts = {writer.frame_count for writer in self._writers.values()}
        if not counts:
            return 0
        if len(counts) != 1:
            raise RuntimeError("multi-view video frame counts are not synchronized")
        return next(iter(counts))

    def write(self, view: str, frame: np.ndarray) -> None:
        self._writers[view].write(frame)

    def __enter__(self):
        try:
            for view, path in self.output_paths.items():
                self._writers[view] = FfmpegVideoWriter(
                    path,
                    self.width,
                    self.height,
                    self.fps,
                )
        except Exception:
            for writer in reversed(tuple(self._writers.values())):
                writer.close()
            raise
        return self

    def __exit__(self, exception_type, _exception, _traceback) -> bool:
        close_error: Exception | None = None
        for writer in reversed(tuple(self._writers.values())):
            try:
                writer.close()
            except Exception as error:
                if close_error is None:
                    close_error = error
        if exception_type is None and close_error is not None:
            raise close_error
        return False
