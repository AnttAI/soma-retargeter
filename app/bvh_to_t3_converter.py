#!/usr/bin/env python3
"""Create T3-ready motion outputs without changing the existing BVH converter.

T3 is the T2 upper body mounted on the two-wheel base.  This command first
retargets BVH motions to the existing T2 CSV format, then derives the
diff-drive wheel CSV used by Kimodo's T3 viewer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = REPO_ROOT / "assets" / "default_bvh_to_csv_converter_config.json"
DEFAULT_T3_EXPORT = REPO_ROOT / "assets" / "motions" / "t3" / "csv"
DEFAULT_WHEEL_EXPORT = REPO_ROOT / "assets" / "motions" / "t3" / "wheel_csv"
DEFAULT_KIMODO_ROOT = Path("/home/jony/Downloads/kimodo")
DEFAULT_T3_VIEWER = DEFAULT_KIMODO_ROOT / "wheel_base_tools" / "view_t3_robot.py"
DEFAULT_T3_URDF = DEFAULT_KIMODO_ROOT / "robot_demo_outputs" / "t3_robot" / "T3.urdf"
DEFAULT_WHEEL_RADIUS_M = 0.10
DEFAULT_WHEEL_SEPARATION_M = 0.38
T3_LIFT_COLUMN = "telescopic_lift_joint_dof"
T3_LIFT_MIN_M = 0.0
T3_LIFT_MAX_M = 0.55
T3_WAIST_HEIGHT_NO_LIFT_M = 0.795
T3_SHOULDER_HEIGHT_NO_LIFT_M = 1.15289
T3_TELESCOPIC_ABSOLUTE_MIN_HEIGHT_M = 0.60
T3_LIFT_HEIGHT_OFFSET_M = -0.03

try:
    from t3_wheel_converter import _stable_path_headings, convert_t2_csv_to_t3_diff_drive
except ModuleNotFoundError:
    from app.t3_wheel_converter import _stable_path_headings, convert_t2_csv_to_t3_diff_drive

try:
    from tqdm import trange
except ModuleNotFoundError:
    def trange(*args, **kwargs):
        return range(*args)

T3_CSV_HEADER = [
    "Frame",
    "root_translateX", "root_translateY", "root_translateZ",
    "root_rotateX", "root_rotateY", "root_rotateZ",
    "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
    "head_pitch_joint_dof", "head_yaw_joint_dof",
    "right_joint1_dof", "right_joint2_dof", "right_joint3_dof",
    "right_joint4_dof", "right_joint5_dof", "right_joint6_dof",
    "right_joint7_dof",
    "right_gripper_joint1_dof", "right_gripper_joint2_dof",
    "left_joint1_dof", "left_joint2_dof", "left_joint3_dof",
    "left_joint4_dof", "left_joint5_dof", "left_joint6_dof",
    "left_joint7_dof",
    "left_gripper_joint1_dof", "left_gripper_joint2_dof",
]
T3_WHEEL_CSV_COLUMNS = [
    "time_s",
    "root_x_m",
    "root_y_m",
    "root_z_m",
    "root_yaw_rad",
    "root_yaw_deg",
    "step_ground_distance_m",
    "distance_from_start_m",
    "forward_velocity_m_s",
    "yaw_rate_rad_s",
    "left_wheel_rad_s",
    "right_wheel_rad_s",
    "left_motor_rpm",
    "right_motor_rpm",
]


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_t3_csv_from_t2_buffer(path: Path, buffer) -> None:
    import soma_retargeter.assets.csv as csv_utils

    t2_config = csv_utils.get_csv_config("t2")
    t2_header = t2_config.csv_header
    t3_indices = [t2_header.index(column) for column in T3_CSV_HEADER]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(T3_CSV_HEADER)
        for frame_idx in range(buffer.num_frames):
            t2_row = t2_config.to_csv_row(frame_idx, buffer.get_data(frame_idx))
            writer.writerow([t2_row[index] for index in t3_indices])


def _compute_bvh_lift_extensions(
    bvh_path: Path,
    source_facing_direction: str,
    match_target: str = "waist",
    lift_height_offset_m: float = T3_LIFT_HEIGHT_OFFSET_M,
    waist_joint_names: tuple[str, ...] = ("Spine1", "Spine2", "Chest"),
    shoulder_joint_names: tuple[str, ...] = ("LeftShoulder", "RightShoulder"),
) -> np.ndarray:
    import warp as wp

    import soma_retargeter.assets.bvh as bvh_utils
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str

    skeleton, animation = bvh_utils.load_bvh(bvh_path)
    converter = SpaceConverter(get_facing_direction_type_from_str(source_facing_direction))
    offset = converter.transform(wp.transform_identity())

    def joint_indices(names: tuple[str, ...]) -> list[int]:
        indices = [skeleton.joint_index(name) for name in names]
        return [idx for idx in indices if idx != -1]

    waist_indices = joint_indices(waist_joint_names)
    shoulder_indices = joint_indices(shoulder_joint_names)
    if not waist_indices and match_target in {"waist", "average"}:
        raise RuntimeError(f"BVH does not expose any waist joints {waist_joint_names!r}: {bvh_path}")
    if not shoulder_indices and match_target in {"waist", "shoulders", "average"}:
        raise RuntimeError(f"BVH does not expose any shoulder joints {shoulder_joint_names!r}: {bvh_path}")

    lift_extensions = np.zeros(animation.num_frames, dtype=np.float64)
    for frame_idx in range(animation.num_frames):
        transforms = animation.compute_global_transforms(frame_idx, offset)
        required_extensions = []
        waist_height = None
        shoulder_height = None
        if match_target in {"waist", "average"}:
            waist_height = float(np.mean([float(transforms[idx][2]) for idx in waist_indices]))
        if match_target in {"waist", "shoulders", "average"}:
            shoulder_height = float(np.mean([float(transforms[idx][2]) for idx in shoulder_indices]))

        if match_target == "waist":
            # Use the waist as the anchor, but shift it by the torso-proportion
            # difference so the T3 shoulders land at the human shoulder height.
            human_waist_to_shoulder = shoulder_height - waist_height
            t3_waist_to_shoulder = T3_SHOULDER_HEIGHT_NO_LIFT_M - T3_WAIST_HEIGHT_NO_LIFT_M
            shoulder_alignment_offset = human_waist_to_shoulder - t3_waist_to_shoulder
            target_waist_height = waist_height + shoulder_alignment_offset
            required_extensions.append(target_waist_height - T3_WAIST_HEIGHT_NO_LIFT_M)
        elif match_target == "shoulders":
            required_extensions.append(shoulder_height - T3_SHOULDER_HEIGHT_NO_LIFT_M)
        elif match_target == "average":
            required_extensions.append(waist_height - T3_WAIST_HEIGHT_NO_LIFT_M)
            required_extensions.append(shoulder_height - T3_SHOULDER_HEIGHT_NO_LIFT_M)
        lift_extensions[frame_idx] = float(np.mean(required_extensions))

    return np.clip(lift_extensions + float(lift_height_offset_m), T3_LIFT_MIN_M, T3_LIFT_MAX_M)


def _append_lift_column_to_t3_csv(t3_csv: Path, lift_extensions_m: np.ndarray) -> None:
    with t3_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{t3_csv} has no header")
        rows = list(reader)

    if not rows:
        return

    original_header = [column for column in reader.fieldnames if column != T3_LIFT_COLUMN]
    insert_idx = original_header.index("head_yaw_joint_dof") + 1 if "head_yaw_joint_dof" in original_header else len(original_header)
    header = original_header[:insert_idx] + [T3_LIFT_COLUMN] + original_header[insert_idx:]

    with t3_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row_idx, row in enumerate(rows):
            out = {column: row.get(column, "") for column in original_header}
            out[T3_LIFT_COLUMN] = float(lift_extensions_m[min(row_idx, len(lift_extensions_m) - 1)])
            writer.writerow(out)
    _print_lift_summary(t3_csv, lift_extensions_m)


def _print_lift_summary(t3_csv: Path, lift_extensions_m: np.ndarray) -> None:
    if lift_extensions_m.size == 0:
        return
    lift_min = float(np.min(lift_extensions_m))
    lift_max = float(np.max(lift_extensions_m))
    lift_first = float(lift_extensions_m[0])
    telescope_min = T3_TELESCOPIC_ABSOLUTE_MIN_HEIGHT_M + lift_min
    telescope_max = T3_TELESCOPIC_ABSOLUTE_MIN_HEIGHT_M + lift_max
    waist_min = T3_WAIST_HEIGHT_NO_LIFT_M + lift_min
    waist_max = T3_WAIST_HEIGHT_NO_LIFT_M + lift_max
    shoulder_min = T3_SHOULDER_HEIGHT_NO_LIFT_M + lift_min
    shoulder_max = T3_SHOULDER_HEIGHT_NO_LIFT_M + lift_max
    print(
        f"[INFO]: Lift column added: {t3_csv}\n"
        f"        extension_m: first={lift_first:.4f}, min={lift_min:.4f}, max={lift_max:.4f}, "
        f"limits=[{T3_LIFT_MIN_M:.2f}, {T3_LIFT_MAX_M:.2f}]\n"
        f"        telescopic_abs_m: min={telescope_min:.4f}, max={telescope_max:.4f}, "
        f"limits=[{T3_TELESCOPIC_ABSOLUTE_MIN_HEIGHT_M:.2f}, {T3_TELESCOPIC_ABSOLUTE_MIN_HEIGHT_M + T3_LIFT_MAX_M:.2f}]\n"
        f"        t3_waist_abs_m: min={waist_min:.4f}, max={waist_max:.4f}\n"
        f"        t3_shoulder_abs_m: min={shoulder_min:.4f}, max={shoulder_max:.4f}"
    )


def _retarget_bvh_to_t3(
    config: dict,
    t3_export: Path,
    lift_match_target: str = "waist",
    lift_height_offset_m: float = T3_LIFT_HEIGHT_OFFSET_M,
    include_lift_column: bool = True,
) -> list[tuple[Path, Path]]:
    import warp as wp

    import soma_retargeter.assets.bvh as bvh_utils
    import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str

    import_root = (REPO_ROOT / config["import_folder"]).resolve()
    if not import_root.exists():
        raise FileNotFoundError(f"BVH import folder not found: {import_root}")

    bvh_paths = sorted(import_root.rglob("*.bvh"))
    if not bvh_paths:
        raise FileNotFoundError(f"No BVH files found under {import_root}")

    batch_size = int(config.get("batch_size", 100))
    converter = SpaceConverter(get_facing_direction_type_from_str(config.get("retarget_source_facing_direction", "Mujoco")))
    bvh_tx_converter = converter.transform(wp.transform_identity())

    bvh_importer = bvh_utils.BVHImporter()
    bvh_skeleton, _ = bvh_importer.create_skeleton(bvh_paths[0])
    expected_num_joints = bvh_skeleton.num_joints

    pipeline = newton_pipeline.NewtonPipeline(
        bvh_skeleton,
        config.get("retarget_source", "soma"),
        "t2")

    outputs: list[tuple[Path, Path]] = []
    batches = [bvh_paths[idx:idx + batch_size] for idx in range(0, len(bvh_paths), batch_size)]
    for batch_idx, batch in enumerate(batches):
        print(f"[INFO]: T3 processing batch {batch_idx + 1} of {len(batches)}")
        animations = []
        for bvh_path in batch:
            _, animation = bvh_utils.load_bvh(bvh_path, bvh_skeleton)
            if animation.skeleton.num_joints != expected_num_joints:
                raise RuntimeError(
                    f"Unexpected number of joints in {bvh_path}. "
                    f"Expected {expected_num_joints}, got {animation.skeleton.num_joints}")
            animations.append(animation)

        pipeline.clear()
        pipeline.add_input_motions(animations, [bvh_tx_converter] * len(animations), True)
        csv_buffers = pipeline.execute()
        if len(csv_buffers) != len(batch):
            raise RuntimeError("Retargeted buffer count does not match input motion count")

        for motion_idx in trange(len(csv_buffers), desc="[INFO]: Exporting T3 CSV Files"):
            bvh_path = batch[motion_idx]
            relative = bvh_path.relative_to(import_root).with_suffix(".csv")
            t3_csv = t3_export / relative
            _save_t3_csv_from_t2_buffer(t3_csv, csv_buffers[motion_idx])
            if include_lift_column:
                lift_extensions = _compute_bvh_lift_extensions(
                    bvh_path,
                    config.get("retarget_source_facing_direction", "Mujoco"),
                    lift_match_target,
                    lift_height_offset_m,
                )
                _append_lift_column_to_t3_csv(t3_csv, lift_extensions)
            outputs.append((bvh_path, t3_csv))

    return outputs


def _write_wheel_csv_from_root_trajectory(
    root_xy_m: np.ndarray,
    yaw_rad: np.ndarray,
    output_csv: Path,
    fps: float,
    wheel_radius_m: float,
    wheel_separation_m: float,
    max_forward_speed: float | None,
    max_yaw_rate: float | None,
    drive_heading_rad: np.ndarray | None = None,
    enforce_no_slip: bool = False,
) -> None:
    root_xy_m = np.asarray(root_xy_m, dtype=np.float64).copy()
    yaw_rad = np.unwrap(np.asarray(yaw_rad, dtype=np.float64))
    if root_xy_m.shape[0] != yaw_rad.shape[0]:
        raise ValueError("root trajectory and yaw must have the same number of frames")
    if root_xy_m.shape[0] < 2:
        raise ValueError("Need at least two frames to compute wheel commands")
    root_xy_m -= root_xy_m[0]

    dt = 1.0 / fps
    ground_delta = root_xy_m[1:] - root_xy_m[:-1]
    drive_heading_rad = yaw_rad if drive_heading_rad is None else np.unwrap(
        np.asarray(drive_heading_rad, dtype=np.float64)
    )
    if drive_heading_rad.shape[0] != root_xy_m.shape[0]:
        raise ValueError("drive heading and root trajectory must have the same number of frames")
    heading = np.stack([np.cos(drive_heading_rad[:-1]), np.sin(drive_heading_rad[:-1])], axis=1)
    forward_step_m = np.sum(ground_delta * heading, axis=1)
    if enforce_no_slip:
        no_slip_xy = np.zeros_like(root_xy_m)
        heading_mid = 0.5 * (drive_heading_rad[:-1] + drive_heading_rad[1:])
        for frame_idx in range(1, root_xy_m.shape[0]):
            distance = forward_step_m[frame_idx - 1]
            no_slip_xy[frame_idx] = no_slip_xy[frame_idx - 1] + distance * np.array(
                [math.cos(heading_mid[frame_idx - 1]), math.sin(heading_mid[frame_idx - 1])],
                dtype=np.float64,
            )
        root_xy_m = no_slip_xy
        ground_delta = root_xy_m[1:] - root_xy_m[:-1]
    step_distance_m = np.linalg.norm(ground_delta, axis=1)
    distance_from_start_m = np.concatenate([[0.0], np.cumsum(step_distance_m)])
    forward_velocity_m_s = forward_step_m / dt
    yaw_rate_rad_s = np.diff(yaw_rad) / dt

    if max_forward_speed is not None:
        forward_velocity_m_s = np.clip(forward_velocity_m_s, -max_forward_speed, max_forward_speed)
    if max_yaw_rate is not None:
        yaw_rate_rad_s = np.clip(yaw_rate_rad_s, -max_yaw_rate, max_yaw_rate)

    # Keep visual wheel rates and motor RPM in the same standard diff-drive
    # convention used by Kimodo's T3 playback.  robot_app applies any real
    # TaraBase wiring signs later when streaming to hardware.
    left_rad_s = (forward_velocity_m_s - yaw_rate_rad_s * wheel_separation_m / 2.0) / wheel_radius_m
    right_rad_s = (forward_velocity_m_s + yaw_rate_rad_s * wheel_separation_m / 2.0) / wheel_radius_m
    rad_s_to_rpm = 60.0 / (2.0 * math.pi)

    def pad_last(values: np.ndarray) -> np.ndarray:
        return np.concatenate([values, values[-1:]], axis=0)

    step_distance_m = np.concatenate([[0.0], step_distance_m])
    forward_velocity_m_s = pad_last(forward_velocity_m_s)
    yaw_rate_rad_s = pad_last(yaw_rate_rad_s)
    left_rad_s = pad_last(left_rad_s)
    right_rad_s = pad_last(right_rad_s)
    left_motor_rpm = np.rint(left_rad_s * rad_s_to_rpm).astype(np.int64)
    right_motor_rpm = np.rint(right_rad_s * rad_s_to_rpm).astype(np.int64)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Frame",
            "root_x_m",
            "root_y_m",
            "root_z_m",
            "root_yaw_rad",
            "root_yaw_deg",
            "step_ground_distance_m",
            "distance_from_start_m",
            "forward_velocity_m_s",
            "yaw_rate_rad_s",
            "left_wheel_rad_s",
            "right_wheel_rad_s",
            "left_motor_rpm",
            "right_motor_rpm",
        ])
        for frame_idx in range(root_xy_m.shape[0]):
            writer.writerow([
                frame_idx,
                root_xy_m[frame_idx, 0],
                root_xy_m[frame_idx, 1],
                0.0,
                yaw_rad[frame_idx],
                math.degrees(yaw_rad[frame_idx]),
                step_distance_m[frame_idx],
                distance_from_start_m[frame_idx],
                forward_velocity_m_s[frame_idx],
                yaw_rate_rad_s[frame_idx],
                left_rad_s[frame_idx],
                right_rad_s[frame_idx],
                int(left_motor_rpm[frame_idx]),
                int(right_motor_rpm[frame_idx]),
            ])


def _append_wheel_columns_to_t3_csv(t3_csv: Path, wheel_csv: Path, fps: float) -> None:
    with t3_csv.open(newline="", encoding="utf-8") as f:
        t3_reader = csv.DictReader(f)
        if t3_reader.fieldnames is None:
            raise ValueError(f"{t3_csv} has no header")
        t3_rows = list(t3_reader)

    with wheel_csv.open(newline="", encoding="utf-8") as f:
        wheel_reader = csv.DictReader(f)
        if wheel_reader.fieldnames is None:
            raise ValueError(f"{wheel_csv} has no header")
        wheel_rows = list(wheel_reader)

    if not t3_rows or not wheel_rows:
        return

    original_header = [column for column in t3_reader.fieldnames if column not in T3_WHEEL_CSV_COLUMNS]
    header = original_header + T3_WHEEL_CSV_COLUMNS

    with t3_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row_idx, t3_row in enumerate(t3_rows):
            wheel_row = wheel_rows[min(row_idx, len(wheel_rows) - 1)]
            out = {column: t3_row.get(column, "") for column in original_header}
            out["time_s"] = row_idx / fps if fps > 0 else 0.0
            for column in T3_WHEEL_CSV_COLUMNS:
                if column == "time_s":
                    continue
                out[column] = wheel_row.get(column, 0.0)
            writer.writerow(out)


def _is_standing_root_motion(root_xy_m: np.ndarray, threshold_m: float) -> bool:
    root_xy_m = np.asarray(root_xy_m, dtype=np.float64)
    if root_xy_m.shape[0] == 0:
        return True
    displacement = np.linalg.norm(root_xy_m - root_xy_m[0], axis=1)
    return float(np.max(displacement)) <= threshold_m


def _clamp_root_xy_radius(root_xy_m: np.ndarray, radius_m: float) -> np.ndarray:
    root_xy_m = np.asarray(root_xy_m, dtype=np.float64).copy()
    if root_xy_m.shape[0] == 0:
        return root_xy_m
    origin = root_xy_m[0].copy()
    delta = root_xy_m - origin
    norms = np.linalg.norm(delta, axis=1)
    scale = np.ones_like(norms)
    mask = norms > radius_m
    scale[mask] = radius_m / np.clip(norms[mask], 1e-8, None)
    return origin + delta * scale[:, None]


def _joint_facing_heading(
    skeleton,
    joint_targets: np.ndarray,
    base_direction_sign: float,
) -> np.ndarray:
    forward_axis = np.asarray(skeleton.forward_axis, dtype=np.float64)
    forward_world = R.from_quat(joint_targets[:, 3:7]).apply(forward_axis)
    heading = np.unwrap(np.arctan2(forward_world[:, 1], forward_world[:, 0]))
    if base_direction_sign < 0.0:
        heading = heading + math.pi
    return np.unwrap(heading)


def _align_heading_with_path(heading: np.ndarray, path_heading: np.ndarray) -> np.ndarray:
    heading = np.unwrap(np.asarray(heading, dtype=np.float64))
    path_heading = np.unwrap(np.asarray(path_heading, dtype=np.float64))
    if heading.shape != path_heading.shape or heading.shape[0] == 0:
        return heading
    if float(np.mean(np.cos(heading - path_heading))) < 0.0:
        heading = heading + math.pi
    return np.unwrap(heading)


def _convert_bvh_hips_to_wheels(
    bvh_path: Path,
    wheel_csv: Path,
    fps: float,
    wheel_radius_m: float,
    wheel_separation_m: float,
    max_forward_speed: float | None,
    max_yaw_rate: float | None,
    source_facing_direction: str,
    base_direction_sign: float,
    standing_motion_threshold: float,
    standing_base_radius: float,
    base_joint_name: str = "Hips",
    base_yaw_source: str = "waist",
    standing_yaw_offset_deg: float = 180.0,
) -> None:
    import warp as wp

    import soma_retargeter.assets.bvh as bvh_utils
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str

    skeleton, animation = bvh_utils.load_bvh(bvh_path)
    base_joint_idx = skeleton.joint_index(base_joint_name)
    if base_joint_idx == -1:
        raise RuntimeError(f"BVH does not expose base joint {base_joint_name!r}: {bvh_path}")

    converter = SpaceConverter(get_facing_direction_type_from_str(source_facing_direction))
    offset = converter.transform(wp.transform_identity())
    base_targets = np.asarray(
        [
            animation.compute_global_transforms(frame_idx, offset)[base_joint_idx]
            for frame_idx in range(animation.num_frames)
        ],
        dtype=np.float64,
    )
    base_pos = base_targets[:, 0:3]
    root_xy_m = base_direction_sign * (base_pos[:, 0:2] - base_pos[0, 0:2])
    waist_heading = _joint_facing_heading(skeleton, base_targets, base_direction_sign)
    standing_motion = _is_standing_root_motion(root_xy_m, standing_motion_threshold)

    path_heading = _stable_path_headings(root_xy_m, fps) if not standing_motion else waist_heading
    if not standing_motion:
        waist_heading = _align_heading_with_path(waist_heading, path_heading)
    else:
        # In-place BVH clips have no travel direction, so the path cannot
        # disambiguate which side of the waist frame should be treated as T3
        # front.  Generated Kimodo standing clips need this half-turn to match
        # the visual human facing in the Newton T3 viewer.
        waist_heading = np.unwrap(waist_heading + math.radians(standing_yaw_offset_deg))

    if base_yaw_source == "waist" or standing_motion:
        base_yaw = waist_heading
        root_xy_m = _clamp_root_xy_radius(root_xy_m, standing_base_radius)
        drive_heading = waist_heading
        if not standing_motion:
            root_xy_m = base_direction_sign * (base_pos[:, 0:2] - base_pos[0, 0:2])
    else:
        drive_heading = path_heading
        # Store the same absolute path heading used to reconstruct the wheel
        # trajectory.  This mirrors Kimodo's wheel-base viewer and avoids the
        # visual/physical mismatch caused by subtracting frame-0 heading.
        base_yaw = np.unwrap(drive_heading)

    _write_wheel_csv_from_root_trajectory(
        root_xy_m=root_xy_m,
        yaw_rad=base_yaw,
        output_csv=wheel_csv,
        fps=fps,
        wheel_radius_m=wheel_radius_m,
        wheel_separation_m=wheel_separation_m,
        max_forward_speed=max_forward_speed,
        max_yaw_rate=max_yaw_rate,
        drive_heading_rad=drive_heading,
        enforce_no_slip=True,
    )


def _convert_t3_to_wheels(
    t3_export: Path,
    wheel_export: Path,
    bvh_import_root: Path | None,
    fps: float,
    wheel_radius_m: float,
    wheel_separation_m: float,
    max_forward_speed: float | None,
    max_yaw_rate: float | None,
    yaw_source: str,
    waist_yaw_compensation: float,
    base_source: str,
    source_facing_direction: str,
    base_direction_sign: float,
    standing_motion_threshold: float,
    standing_base_radius: float,
    base_joint_name: str,
    base_yaw_source: str,
    standing_yaw_offset_deg: float,
) -> list[tuple[Path, Path]]:
    t3_paths = sorted(t3_export.rglob("*.csv"))
    if not t3_paths:
        raise FileNotFoundError(f"No T3 CSV files found under {t3_export}")

    outputs: list[tuple[Path, Path]] = []
    for t3_csv in t3_paths:
        relative = t3_csv.relative_to(t3_export)
        wheel_csv = (wheel_export / relative).with_name(f"{relative.stem}_diff_drive.csv")
        converted_from_bvh = False
        if base_source == "human-hips" and bvh_import_root is not None:
            bvh_path = (bvh_import_root / relative).with_suffix(".bvh")
            if bvh_path.exists():
                _convert_bvh_hips_to_wheels(
                    bvh_path=bvh_path,
                    wheel_csv=wheel_csv,
                    fps=fps,
                    wheel_radius_m=wheel_radius_m,
                    wheel_separation_m=wheel_separation_m,
                    max_forward_speed=max_forward_speed,
                    max_yaw_rate=max_yaw_rate,
                    source_facing_direction=source_facing_direction,
                    base_direction_sign=base_direction_sign,
                    standing_motion_threshold=standing_motion_threshold,
                    standing_base_radius=standing_base_radius,
                    base_joint_name=base_joint_name,
                    base_yaw_source=base_yaw_source,
                    standing_yaw_offset_deg=standing_yaw_offset_deg,
                )
                converted_from_bvh = True

        if not converted_from_bvh:
            convert_t2_csv_to_t3_diff_drive(
                input_csv=t3_csv,
                output_csv=wheel_csv,
                fps=fps,
                wheel_radius_m=wheel_radius_m,
                wheel_separation_m=wheel_separation_m,
                max_forward_speed=max_forward_speed,
                max_yaw_rate=max_yaw_rate,
                yaw_source=yaw_source,
                waist_yaw_compensation=waist_yaw_compensation,
            )
        _append_wheel_columns_to_t3_csv(t3_csv, wheel_csv, fps)
        outputs.append((t3_csv, wheel_csv))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget BVH motions into T3 upper-body + wheel CSV outputs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Existing BVH converter config to reuse.")
    parser.add_argument("--t3-export-folder", type=Path, default=DEFAULT_T3_EXPORT, help="Output folder for T3 upper-body CSVs.")
    parser.add_argument("--t2-export-folder", dest="t3_export_folder", type=Path, help="Deprecated alias for --t3-export-folder.")
    parser.add_argument("--wheel-export-folder", type=Path, default=DEFAULT_WHEEL_EXPORT, help="Output folder for wheel CSVs.")
    parser.add_argument("--kimodo-root", type=Path, default=DEFAULT_KIMODO_ROOT, help="Kimodo checkout root.")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate for wheel command generation.")
    parser.add_argument("--wheel-radius", type=float, default=DEFAULT_WHEEL_RADIUS_M, help="Wheel radius in meters.")
    parser.add_argument("--wheel-separation", type=float, default=DEFAULT_WHEEL_SEPARATION_M, help="Wheel separation in meters.")
    parser.add_argument("--max-forward-speed", type=float, default=3.0, help="Optional forward speed clamp in m/s.")
    parser.add_argument("--max-yaw-rate", type=float, default=12.0, help="Optional yaw-rate clamp in rad/s.")
    parser.add_argument("--yaw-source", choices=("auto", "path", "t2"), default="auto", help="Wheel yaw source.")
    parser.add_argument("--waist-yaw-compensation", type=float, default=1.0, help="How much T2 waist yaw to add to T3 wheel-base yaw. Use 0 to disable.")
    parser.add_argument("--base-source", choices=("human-hips", "t2-root"), default="human-hips", help="Generate T3 base motion from matching BVH human hips or from the retargeted T2 root.")
    parser.add_argument("--base-direction-sign", type=float, choices=(-1.0, 1.0), default=1.0, help="BVH hips trajectory direction. +1 preserves the rendered SOMA human world direction.")
    parser.add_argument("--base-joint", default="Hips", help="BVH joint used as the T3 base/waist source. Default: Hips.")
    parser.add_argument("--base-yaw-source", choices=("waist", "path"), default="waist", help="Use the BVH waist facing rotation or ground path tangent for T3 base yaw.")
    parser.add_argument("--standing-yaw-offset-deg", type=float, default=180.0, help="Extra yaw applied only to standing/in-place human-base clips.")
    parser.add_argument("--standing-motion-threshold", type=float, default=0.25, help="Treat clips with root displacement below this many meters as standing motions.")
    parser.add_argument("--standing-base-radius", type=float, default=0.04, help="Maximum XY base translation radius for standing motions, in meters.")
    parser.add_argument(
        "--lift-match-target",
        choices=("average", "waist", "shoulders"),
        default="waist",
        help=(
            "Generate telescopic_lift_joint_dof from BVH height. "
            "waist anchors the waist plus a shoulder-alignment offset; "
            "average balances waist and shoulders; shoulders matches only shoulders."
        ),
    )
    parser.add_argument(
        "--lift-height-offset-m",
        type=float,
        default=T3_LIFT_HEIGHT_OFFSET_M,
        help="Extra lift calibration in meters. Negative lowers T3; default -0.03 lowers it 3 cm.",
    )
    parser.add_argument("--no-lift-column", action="store_true", help="Do not add telescopic_lift_joint_dof to T3 CSVs.")
    parser.add_argument("--skip-retarget", action="store_true", help="Only convert existing T3 upper-body CSVs to wheel CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    t3_export = args.t3_export_folder.expanduser().resolve()
    wheel_export = args.wheel_export_folder.expanduser().resolve()
    kimodo_root = args.kimodo_root.expanduser().resolve()
    t3_viewer = kimodo_root / "wheel_base_tools" / "view_t3_robot.py"
    t3_urdf = kimodo_root / "robot_demo_outputs" / "t3_robot" / "T3.urdf"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = _load_json(config_path)
    bvh_import_root = (REPO_ROOT / config["import_folder"]).resolve() if "import_folder" in config else None
    source_facing_direction = config.get("retarget_source_facing_direction", "Mujoco")

    if not args.skip_retarget:
        _retarget_bvh_to_t3(
            config,
            t3_export,
            lift_match_target=args.lift_match_target,
            lift_height_offset_m=args.lift_height_offset_m,
            include_lift_column=not args.no_lift_column,
        )

    outputs = _convert_t3_to_wheels(
        t3_export=t3_export,
        wheel_export=wheel_export,
        bvh_import_root=bvh_import_root,
        fps=args.fps,
        wheel_radius_m=args.wheel_radius,
        wheel_separation_m=args.wheel_separation,
        max_forward_speed=args.max_forward_speed,
        max_yaw_rate=args.max_yaw_rate,
        yaw_source=args.yaw_source,
        waist_yaw_compensation=args.waist_yaw_compensation,
        base_source=args.base_source,
        source_facing_direction=source_facing_direction,
        base_direction_sign=args.base_direction_sign,
        standing_motion_threshold=args.standing_motion_threshold,
        standing_base_radius=args.standing_base_radius,
        base_joint_name=args.base_joint,
        base_yaw_source=args.base_yaw_source,
        standing_yaw_offset_deg=args.standing_yaw_offset_deg,
    )

    print(f"[INFO]: T3 upper-body CSV folder: {t3_export}")
    print(f"[INFO]: T3 wheel CSV folder: {wheel_export}")
    print(f"[INFO]: Generated wheel CSVs: {len(outputs)}")
    if t3_viewer.exists():
        print("[INFO]: View with stiff T3 posture:")
        print(
            "  "
            f"{sys.executable} {t3_viewer} "
            f"--t3-urdf {t3_urdf} "
            f"--t2-csv-root {t3_export} "
            f"--wheel-csv-root {wheel_export} "
            f"--fps {args.fps}"
        )


if __name__ == "__main__":
    main()
