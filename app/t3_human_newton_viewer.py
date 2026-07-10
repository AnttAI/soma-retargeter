#!/usr/bin/env python3
"""Newton viewer for a SOMA human mesh beside the T3 robot."""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import newton
import newton.ik as ik
import warp as wp
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from viewer_compat import enable_cpu_pinned_fallback
except ModuleNotFoundError:
    from app.viewer_compat import enable_cpu_pinned_fallback

try:
    from cpu_robot_mesh_renderer import CpuRobotMeshRenderer
except ModuleNotFoundError:
    from app.cpu_robot_mesh_renderer import CpuRobotMeshRenderer

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.assets.csv as csv_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.pipelines.utils as pipeline_utils
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
from soma_retargeter.renderers.coordinate_renderer import CoordinateRenderer
from soma_retargeter.renderers.mesh_renderer import SkeletalMeshRenderer
from soma_retargeter.renderers.skeleton_renderer import SkeletonRenderer
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType

try:
    from t3_wheel_converter import _stable_path_headings, convert_t2_csv_to_t3_diff_drive
except ModuleNotFoundError:
    from app.t3_wheel_converter import _stable_path_headings, convert_t2_csv_to_t3_diff_drive


DEFAULT_KIMODO_ROOT = Path("/home/jony/Downloads/kimodo")
DEFAULT_T3_URDF = Path("/home/jony/Downloads/kimodo/robot_demo_outputs/t3_robot/T3.urdf")
DEFAULT_T3_SCALE = 1.0
T3_LEFT_SHOULDER_HEIGHT_M = 1.51377988
T3_RIGHT_SHOULDER_HEIGHT_M = 1.51374988
T3_SHOULDER_HEIGHT_M = 0.5 * (T3_LEFT_SHOULDER_HEIGHT_M + T3_RIGHT_SHOULDER_HEIGHT_M)
_UI_PANEL_WIDTH = 320
_UI_PANEL_MARGIN = 10
_UI_PANEL_ALPHA = 0.9

T2_TO_T3_JOINTS = {
    "waist_yaw_joint_dof": "waist_yaw_joint",
    "waist_roll_joint_dof": "waist_roll_joint",
    "waist_pitch_joint_dof": "waist_pitch_joint",
    "head_pitch_joint_dof": "head_pitch_joint",
    "head_yaw_joint_dof": "head_yaw_joint",
    "right_joint1_dof": "right_joint1",
    "right_joint2_dof": "right_joint2",
    "right_joint3_dof": "right_joint3",
    "right_joint4_dof": "right_joint4",
    "right_joint5_dof": "right_joint5",
    "right_joint6_dof": "right_joint6",
    "right_joint7_dof": "right_joint7",
    "right_gripper_joint1_dof": "right_gripper_joint1",
    "right_gripper_joint2_dof": "right_gripper_joint2",
    "left_joint1_dof": "left_joint1",
    "left_joint2_dof": "left_joint2",
    "left_joint3_dof": "left_joint3",
    "left_joint4_dof": "left_joint4",
    "left_joint5_dof": "left_joint5",
    "left_joint6_dof": "left_joint6",
    "left_joint7_dof": "left_joint7",
    "left_gripper_joint1_dof": "left_gripper_joint1",
    "left_gripper_joint2_dof": "left_gripper_joint2",
}
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
T3_STIFF_POSTURE_JOINTS = {
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
}
T3_NEUTRAL_JOINTS = {}
T3_IK_MAP = {
    "Chest": ("torso_link", 1.0, 0.15),
    "LeftArm": ("left_link2", 3.0, 0.05),
    "LeftForeArm": ("left_link4", 7.0, 0.0),
    "LeftHand": ("left_link7", 24.0, 0.6),
    "RightArm": ("right_link2", 3.0, 0.05),
    "RightForeArm": ("right_link4", 7.0, 0.0),
    "RightHand": ("right_link7", 24.0, 0.6),
}


@dataclass
class T2Motion:
    sample_rate: float
    joint_angles: dict[str, np.ndarray]
    root_pos: np.ndarray | None = None
    root_quat: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        if self.joint_angles:
            return len(next(iter(self.joint_angles.values())))
        return len(self.root_pos) if self.root_pos is not None else 0

    @property
    def duration(self) -> float:
        return self.num_frames / self.sample_rate if self.sample_rate > 0.0 else 0.0

    def frame_index(self, time_s: float) -> int:
        if self.num_frames == 0:
            return 0
        return max(0, min(int(time_s * self.sample_rate), self.num_frames - 1))


@dataclass
class WheelMotion:
    sample_rate: float
    x_m: np.ndarray
    y_m: np.ndarray
    yaw_rad: np.ndarray
    left_angle: np.ndarray
    right_angle: np.ndarray

    @property
    def num_frames(self) -> int:
        return len(self.x_m)

    @property
    def duration(self) -> float:
        return self.num_frames / self.sample_rate if self.sample_rate > 0.0 else 0.0

    def frame_index(self, time_s: float) -> int:
        if self.num_frames == 0:
            return 0
        return max(0, min(int(time_s * self.sample_rate), self.num_frames - 1))


def _fixed_wheel_motion(num_frames: int, fps: float) -> WheelMotion:
    num_frames = max(1, int(num_frames))
    zeros = np.zeros(num_frames, dtype=np.float64)
    return WheelMotion(
        sample_rate=fps,
        x_m=zeros.copy(),
        y_m=zeros.copy(),
        yaw_rad=zeros.copy(),
        left_angle=zeros.copy(),
        right_angle=zeros.copy(),
    )


def _wheel_motion_from_root_trajectory(
    root_xy_m: np.ndarray,
    yaw_rad: np.ndarray,
    fps: float,
    wheel_radius_m: float,
    wheel_separation_m: float,
    drive_heading_rad: np.ndarray | None = None,
    enforce_no_slip: bool = False,
) -> WheelMotion:
    root_xy_m = np.asarray(root_xy_m, dtype=np.float64).copy()
    yaw_rad = np.unwrap(np.asarray(yaw_rad, dtype=np.float64))
    if root_xy_m.shape[0] != yaw_rad.shape[0]:
        raise ValueError("root trajectory and yaw must have the same number of frames")
    if root_xy_m.shape[0] > 0:
        root_xy_m -= root_xy_m[0]

    num_frames = max(1, yaw_rad.shape[0])
    left_angle = np.zeros(num_frames, dtype=np.float64)
    right_angle = np.zeros(num_frames, dtype=np.float64)
    if num_frames < 2:
        return WheelMotion(fps, root_xy_m[:, 0].copy(), root_xy_m[:, 1].copy(), yaw_rad.copy(), left_angle, right_angle)

    dt = 1.0 / fps
    ground_delta = root_xy_m[1:] - root_xy_m[:-1]
    drive_heading_rad = yaw_rad if drive_heading_rad is None else np.unwrap(
        np.asarray(drive_heading_rad, dtype=np.float64)
    )
    if drive_heading_rad.shape[0] != num_frames:
        raise ValueError("drive heading and root trajectory must have the same number of frames")
    heading = np.stack([np.cos(drive_heading_rad[:-1]), np.sin(drive_heading_rad[:-1])], axis=1)
    forward_step_m = np.sum(ground_delta * heading, axis=1)
    if enforce_no_slip:
        # A differential-drive base has no lateral degree of freedom. Rebuild
        # the displayed root from center-wheel travel and heading so wheel
        # rotation and chassis translation obey the same rolling constraint.
        no_slip_xy = np.zeros_like(root_xy_m)
        heading_mid = 0.5 * (drive_heading_rad[:-1] + drive_heading_rad[1:])
        for frame_idx in range(1, num_frames):
            distance = forward_step_m[frame_idx - 1]
            no_slip_xy[frame_idx] = no_slip_xy[frame_idx - 1] + distance * np.array(
                [math.cos(heading_mid[frame_idx - 1]), math.sin(heading_mid[frame_idx - 1])],
                dtype=np.float64,
            )
        root_xy_m = no_slip_xy
    forward_velocity_m_s = forward_step_m / dt
    yaw_rate_rad_s = np.diff(yaw_rad) / dt

    left_rad_s = (forward_velocity_m_s - yaw_rate_rad_s * wheel_separation_m / 2.0) / wheel_radius_m
    right_rad_s = (forward_velocity_m_s + yaw_rate_rad_s * wheel_separation_m / 2.0) / wheel_radius_m
    for frame_idx in range(1, num_frames):
        left_angle[frame_idx] = left_angle[frame_idx - 1] + left_rad_s[frame_idx - 1] * dt
        right_angle[frame_idx] = right_angle[frame_idx - 1] + right_rad_s[frame_idx - 1] * dt

    return WheelMotion(
        sample_rate=fps,
        x_m=root_xy_m[:, 0].copy(),
        y_m=root_xy_m[:, 1].copy(),
        yaw_rad=yaw_rad.copy(),
        left_angle=left_angle,
        right_angle=right_angle,
    )


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


def _joint_facing_heading(skeleton, joint_targets: np.ndarray, base_direction_sign: float) -> np.ndarray:
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


def _read_t2_motion(path: Path, fps: float) -> T2Motion:
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)

    joint_angles: dict[str, np.ndarray] = {}
    for column_name, joint_name in T2_TO_T3_JOINTS.items():
        if column_name in header:
            joint_angles[joint_name] = np.deg2rad(data[:, header.index(column_name)]).astype(np.float64)

    root_pos = np.zeros((data.shape[0], 3), dtype=np.float64)
    root_quat = np.zeros((data.shape[0], 4), dtype=np.float64)
    root_quat[:, 3] = 1.0
    root_translate_columns = ("root_translateX", "root_translateY", "root_translateZ")
    root_rotate_columns = ("root_rotateX", "root_rotateY", "root_rotateZ")
    if all(column in header for column in root_translate_columns):
        root_pos[:, 0] = data[:, header.index("root_translateX")] * 0.01
        root_pos[:, 1] = data[:, header.index("root_translateY")] * 0.01
        root_pos[:, 2] = data[:, header.index("root_translateZ")] * 0.01
    if all(column in header for column in root_rotate_columns):
        root_quat[:] = R.from_euler(
            "xyz",
            data[:, [header.index(column) for column in root_rotate_columns]],
            degrees=True).as_quat()

    return T2Motion(sample_rate=fps, joint_angles=joint_angles, root_pos=root_pos, root_quat=root_quat)


def _is_full_t2_csv(path: Path) -> bool:
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    return all(column in header for column in csv_utils.get_csv_config("t2").csv_header)


def _has_wheel_motion_columns(path: Path) -> bool:
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    required = ("root_x_m", "root_y_m", "root_yaw_rad")
    has_wheel_speeds = (
        all(column in header for column in ("left_motor_rpm", "right_motor_rpm"))
        or all(column in header for column in ("left_wheel_rad_s", "right_wheel_rad_s"))
    )
    return all(column in header for column in required) and has_wheel_speeds


def _wheel_csv_row_values(
    wheel_motion: WheelMotion | None,
    frame_idx: int,
    fps: float,
    wheel_radius_m: float,
    wheel_separation_m: float = 0.38,
) -> list[float | int]:
    if wheel_motion is None or wheel_motion.num_frames == 0:
        return [frame_idx / fps if fps > 0 else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0]

    idx = min(max(frame_idx, 0), wheel_motion.num_frames - 1)
    xy = np.stack([wheel_motion.x_m, wheel_motion.y_m], axis=1)
    if idx > 0:
        step_ground_distance_m = float(np.linalg.norm(xy[idx] - xy[idx - 1]))
        distance_from_start_m = float(np.sum(np.linalg.norm(np.diff(xy[:idx + 1], axis=0), axis=1)))
    else:
        step_ground_distance_m = 0.0
        distance_from_start_m = 0.0
    dt = 1.0 / fps if fps > 0 else 0.0
    if dt > 0.0 and wheel_motion.num_frames > 1:
        if idx < wheel_motion.num_frames - 1:
            src_idx = idx
        else:
            src_idx = idx - 1
        left_rad_s = float((wheel_motion.left_angle[src_idx + 1] - wheel_motion.left_angle[src_idx]) / dt)
        right_rad_s = float((wheel_motion.right_angle[src_idx + 1] - wheel_motion.right_angle[src_idx]) / dt)
    else:
        left_rad_s = 0.0
        right_rad_s = 0.0

    left_linear_m_s = left_rad_s * wheel_radius_m
    right_linear_m_s = right_rad_s * wheel_radius_m
    forward_velocity_m_s = 0.5 * (left_linear_m_s + right_linear_m_s)
    yaw_rate_rad_s = (right_linear_m_s - left_linear_m_s) / wheel_separation_m
    rad_s_to_rpm = 60.0 / (2.0 * math.pi)
    yaw_rad = float(wheel_motion.yaw_rad[idx])
    return [
        frame_idx / fps if fps > 0 else 0.0,
        float(wheel_motion.x_m[idx]),
        float(wheel_motion.y_m[idx]),
        0.0,
        yaw_rad,
        math.degrees(yaw_rad),
        step_ground_distance_m,
        distance_from_start_m,
        forward_velocity_m_s,
        yaw_rate_rad_s,
        left_rad_s,
        right_rad_s,
        int(round(left_rad_s * rad_s_to_rpm)),
        int(round(right_rad_s * rad_s_to_rpm)),
    ]


def _save_t3_csv_from_buffer(
    path: Path,
    buffer,
    wheel_motion: WheelMotion | None = None,
    wheel_radius_m: float = 0.10,
) -> None:
    t2_config = csv_utils.get_csv_config("t2")
    t2_header = t2_config.csv_header
    t3_indices = [t2_header.index(column) for column in T3_CSV_HEADER]
    fps = float(getattr(buffer, "sample_rate", 30.0))
    header = T3_CSV_HEADER + (T3_WHEEL_CSV_COLUMNS if wheel_motion is not None else [])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for frame_idx in range(buffer.num_frames):
            t2_row = t2_config.to_csv_row(frame_idx, buffer.get_data(frame_idx))
            row = [t2_row[index] for index in t3_indices]
            if wheel_motion is not None:
                row.extend(_wheel_csv_row_values(wheel_motion, frame_idx, fps, wheel_radius_m))
            writer.writerow(row)


def _save_t3_csv_from_motion(
    path: Path,
    motion: T2Motion,
    wheel_motion: WheelMotion | None = None,
    wheel_radius_m: float = 0.10,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = T3_CSV_HEADER + (T3_WHEEL_CSV_COLUMNS if wheel_motion is not None else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for frame_idx in range(motion.num_frames):
            if motion.root_pos is not None and motion.root_quat is not None:
                pos_cm = motion.root_pos[frame_idx] * 100.0
                euler_deg = R.from_quat(motion.root_quat[frame_idx]).as_euler("xyz", degrees=True)
                row = [frame_idx, pos_cm[0], pos_cm[1], pos_cm[2], euler_deg[0], euler_deg[1], euler_deg[2]]
            else:
                row = [frame_idx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for column_name in T3_CSV_HEADER[7:]:
                joint_name = T2_TO_T3_JOINTS[column_name]
                values = motion.joint_angles.get(joint_name)
                value = float(values[frame_idx]) if values is not None else 0.0
                row.append(math.degrees(value))
            if wheel_motion is not None:
                row.extend(_wheel_csv_row_values(wheel_motion, frame_idx, motion.sample_rate, wheel_radius_m))
            writer.writerow(row)


def _save_wheel_motion_csv(
    path: Path,
    wheel_motion: WheelMotion,
    wheel_radius_m: float = 0.10,
) -> None:
    """Save the exact wheel trajectory currently shown in the viewer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(T3_WHEEL_CSV_COLUMNS)
        for frame_idx in range(wheel_motion.num_frames):
            writer.writerow(
                _wheel_csv_row_values(
                    wheel_motion,
                    frame_idx,
                    wheel_motion.sample_rate,
                    wheel_radius_m,
                )
            )


def _t2_buffer_to_motion(buffer, sample_rate: float) -> T2Motion:
    joint_angles: dict[str, list[float]] = {joint_name: [] for joint_name in T2_TO_T3_JOINTS.values()}
    root_pos_values: list[np.ndarray] = []
    root_quat_values: list[np.ndarray] = []
    header = csv_utils.get_csv_config("t2").csv_header
    for frame_idx in range(buffer.num_frames):
        data = buffer.get_data(frame_idx)
        root_pos_values.append(np.asarray(data[0:3], dtype=np.float64))
        root_quat_values.append(np.asarray(data[3:7], dtype=np.float64))
        for column_name, joint_name in T2_TO_T3_JOINTS.items():
            column_idx = header.index(column_name)
            # Buffer data omits the Frame column and stores joints in radians.
            joint_angles[joint_name].append(float(data[column_idx]))
    return T2Motion(
        sample_rate=sample_rate,
        joint_angles={name: np.asarray(values, dtype=np.float64) for name, values in joint_angles.items()},
        root_pos=np.asarray(root_pos_values, dtype=np.float64),
        root_quat=np.asarray(root_quat_values, dtype=np.float64),
    )


def _read_wheel_motion(path: Path, fps: float, wheel_radius_m: float) -> WheelMotion:
    with path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)

    def column(name: str) -> np.ndarray:
        return data[:, header.index(name)].astype(np.float64)

    reference_x_m = column("root_x_m")
    reference_y_m = column("root_y_m")
    reference_yaw_rad = np.unwrap(column("root_yaw_rad"))
    dt = 1.0 / fps
    num_frames = data.shape[0]
    left_angle = np.zeros(num_frames, dtype=np.float64)
    right_angle = np.zeros(num_frames, dtype=np.float64)

    # Prefer the lossless floating-point rates when both representations are
    # present. Integer motor RPM is still exported for hardware consumption,
    # but using it for viewer reload introduces cumulative angle drift.
    if "left_wheel_rad_s" in header and "right_wheel_rad_s" in header:
        left_rad_s = column("left_wheel_rad_s")
        right_rad_s = column("right_wheel_rad_s")
    elif "left_motor_rpm" in header and "right_motor_rpm" in header:
        rpm_to_rad_s = 2.0 * math.pi / 60.0
        left_rad_s = column("left_motor_rpm") * rpm_to_rad_s
        right_rad_s = column("right_motor_rpm") * rpm_to_rad_s
    else:
        raise ValueError(
            f"{path} needs left/right wheel rad/s or left/right motor RPM columns"
        )

    if "forward_velocity_m_s" in header and "yaw_rate_rad_s" in header:
        forward_velocity_m_s = column("forward_velocity_m_s")
        yaw_rate_rad_s = column("yaw_rate_rad_s")
    else:
        wheel_separation_m = 0.38
        left_linear_m_s = left_rad_s * wheel_radius_m
        right_linear_m_s = right_rad_s * wheel_radius_m
        forward_velocity_m_s = 0.5 * (left_linear_m_s + right_linear_m_s)
        yaw_rate_rad_s = (right_linear_m_s - left_linear_m_s) / wheel_separation_m

    # Drive the displayed base from the wheel/diff-drive commands, not by
    # blindly copying every saved root pose.  The root columns provide the
    # initial pose and a reference trajectory; integrating the commands here
    # makes Newton expose the same wheel-sign/yaw-convention problems that a
    # wheel-only Viser or hardware player would expose.
    x_m = np.zeros(num_frames, dtype=np.float64)
    y_m = np.zeros(num_frames, dtype=np.float64)
    yaw_rad = np.zeros(num_frames, dtype=np.float64)
    x_m[0] = reference_x_m[0]
    y_m[0] = reference_y_m[0]
    yaw_rad[0] = reference_yaw_rad[0]

    for i in range(1, num_frames):
        omega = yaw_rate_rad_s[i - 1]
        yaw_mid = yaw_rad[i - 1] + 0.5 * omega * dt
        x_m[i] = x_m[i - 1] + forward_velocity_m_s[i - 1] * math.cos(yaw_mid) * dt
        y_m[i] = y_m[i - 1] + forward_velocity_m_s[i - 1] * math.sin(yaw_mid) * dt
        yaw_rad[i] = yaw_rad[i - 1] + omega * dt
        left_angle[i] = left_angle[i - 1] + left_rad_s[i - 1] * dt
        right_angle[i] = right_angle[i - 1] + right_rad_s[i - 1] * dt

    if num_frames > 1:
        pos_err = np.linalg.norm(
            np.stack([x_m - reference_x_m, y_m - reference_y_m], axis=1),
            axis=1,
        )
        yaw_err = np.abs(np.angle(np.exp(1j * (yaw_rad - reference_yaw_rad))))
        if float(np.max(pos_err)) > 0.05 or float(np.max(yaw_err)) > math.radians(5.0):
            print(
                "[WARN]: Wheel commands do not reproduce saved root trajectory "
                f"for {path.name}: max position error {float(np.max(pos_err)):.3f} m, "
                f"max yaw error {math.degrees(float(np.max(yaw_err))):.1f} deg"
            )
    return WheelMotion(fps, x_m, y_m, yaw_rad, left_angle, right_angle)


def _generate_wheel_csv(
    t2_csv: Path,
    wheel_csv: Path,
    fps: float,
    wheel_radius_m: float,
    waist_yaw_compensation: float,
) -> None:
    convert_t2_csv_to_t3_diff_drive(
        input_csv=t2_csv,
        output_csv=wheel_csv,
        fps=fps,
        wheel_radius_m=wheel_radius_m,
        wheel_separation_m=0.38,
        max_forward_speed=3.0,
        max_yaw_rate=12.0,
        waist_yaw_compensation=waist_yaw_compensation,
    )


def _yaw_quat_z(angle_rad: float) -> wp.quat:
    half = 0.5 * angle_rad
    return wp.quat(0.0, 0.0, math.sin(half), math.cos(half))


def _transform_to_numpy(tx: wp.transform) -> np.ndarray:
    return np.array([tx.p[0], tx.p[1], tx.p[2], tx.q[0], tx.q[1], tx.q[2], tx.q[3]], dtype=np.float32)


def _transform_point_np(tx: wp.transform, point: np.ndarray) -> np.ndarray:
    p = wp.transform_point(tx, wp.vec3(float(point[0]), float(point[1]), float(point[2])))
    return np.array([p[0], p[1], p[2]], dtype=np.float64)


def _name_from_label(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _build_label_start_map(labels, starts) -> dict[str, int]:
    return {_name_from_label(label): int(start) for label, start in zip(labels, starts)}


class T3HumanNewtonViewer:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.viewer.vsync = True
        self.args = args
        self.time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.playback_time = 0.0
        self.playback_speed = 1.0
        self.playback_loop = True
        self.is_playing = False
        self.show_human_mesh = True
        self.show_human_skeleton = False
        self.show_joint_axes = False
        self.show_gizmos = True
        self.current_bvh = None
        self.current_t2_csv = None
        self.current_wheel_csv = None
        self.generated_wheel_dir = REPO_ROOT / "assets" / "motions" / "t3" / "wheel_csv"
        self.t3_scale = float(args.t3_scale)
        self.base_mode = args.base_mode

        self.t2_buffer = None
        self.t2_motion = None
        self.wheel_motion = None
        self.playback_total_time = 0.0

        self.model = None
        self.state = None
        self.joint_q = None
        self.joint_q_start = {}

        self.converter = SpaceConverter(FacingDirectionType.MUJOCO)
        self.coordinate_renderer = CoordinateRenderer()
        self.skeleton = None
        self.animation = None
        self.skeleton_instance = None
        self.skeleton_renderer = None
        self.skeletal_mesh = None
        self.skeletal_mesh_renderer = None
        self.human_display_origin = wp.transform_identity()
        self.human_offset = wp.transform(wp.vec3(args.human_x, args.human_y, 0.0), wp.quat_identity())
        self.t3_offset = wp.transform(wp.vec3(args.t3_x, args.t3_y, args.t3_z), wp.quat_identity())
        self.t3_gizmo_pivot = np.array(
            [args.robot_gizmo_pivot_x, args.robot_gizmo_pivot_y, args.robot_gizmo_pivot_z],
            dtype=np.float64,
        )
        self.t3_gizmo_offset = self._root_to_gizmo_transform(self.t3_offset)
        self.t3_current_root_tx = wp.transform(self.t3_offset)
        self.default_human_offset = wp.transform(self.human_offset)
        self.default_t3_offset = wp.transform(self.t3_offset)
        self.default_t3_gizmo_offset = wp.transform(self.t3_gizmo_offset)

        enable_cpu_pinned_fallback(self.viewer)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.set_title("BVH to CSV Converter - t3")
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="free")
        self.cpu_robot_mesh_renderer = None
        self._build_t3_model(self.t3_scale)
        self.viewer.set_world_offsets([0, 0, 0])
        self.viewer.set_camera(wp.vec3(0.0, -6.0, 1.8), 0.0, 90.0)

        self._apply_t3_frame()

    def _root_to_gizmo_transform(self, root_tx: wp.transform) -> wp.transform:
        pivot_world = _transform_point_np(root_tx, self.t3_gizmo_pivot)
        return wp.transform(wp.vec3(*pivot_world), root_tx.q)

    def _gizmo_to_root_transform(self, gizmo_tx: wp.transform) -> wp.transform:
        rotated_pivot = wp.quat_rotate(
            gizmo_tx.q,
            wp.vec3(float(self.t3_gizmo_pivot[0]), float(self.t3_gizmo_pivot[1]), float(self.t3_gizmo_pivot[2])),
        )
        root_pos = wp.vec3(
            gizmo_tx.p[0] - rotated_pivot[0],
            gizmo_tx.p[1] - rotated_pivot[1],
            gizmo_tx.p[2] - rotated_pivot[2],
        )
        return wp.transform(root_pos, gizmo_tx.q)

    def _build_t3_model(self, scale: float) -> None:
        self.t3_scale = float(scale)
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_urdf(str(self.args.t3_urdf), floating=True, scale=self.t3_scale)
        self.model = builder.finalize()
        self.state = self.model.state()
        self.joint_q = self.model.joint_q.numpy().copy()
        self.joint_q_start = {
            label.rsplit("/", 1)[-1]: int(start)
            for label, start in zip(self.model.joint_label, self.model.joint_q_start.numpy())
        }
        self.viewer.set_model(self.model)
        self.cpu_robot_mesh_renderer = None
        if isinstance(self.viewer, newton.viewer.ViewerGL) and not self.viewer.device.is_cuda:
            self.cpu_robot_mesh_renderer = CpuRobotMeshRenderer(self.viewer, self.model)

    def _human_shoulder_height_at_start(self) -> float | None:
        if self.skeleton is None or self.animation is None or self.skeleton_instance is None:
            return None
        self.skeleton_instance.set_local_transforms(self.animation.sample(0.0))
        transforms = self.skeleton_instance.compute_global_transforms()
        heights = []
        for joint_name in ("LeftShoulder", "RightShoulder"):
            joint_idx = self.skeleton.joint_index(joint_name)
            if joint_idx != -1:
                heights.append(float(transforms[joint_idx][2]))
        if not heights:
            return None
        return sum(heights) / len(heights)

    def _compute_human_display_origin(self) -> wp.transform:
        if self.skeleton is None or self.animation is None or self.skeleton_instance is None:
            return wp.transform_identity()

        self.skeleton_instance.set_local_transforms(self.animation.sample(0.0))
        transforms = self.skeleton_instance.compute_global_transforms()
        joint_idx = self.skeleton.joint_index("Hips")
        if joint_idx == -1:
            joint_idx = 0

        p = transforms[joint_idx]
        return wp.transform(wp.vec3(-float(p[0]), -float(p[1]), 0.0), wp.quat_identity())

    def _match_t3_scale_to_human_shoulders(self) -> None:
        human_shoulder_height = self._human_shoulder_height_at_start()
        if human_shoulder_height is None:
            return
        target_scale = human_shoulder_height / T3_SHOULDER_HEIGHT_M
        if abs(target_scale - self.t3_scale) < 1e-5:
            return
        self._build_t3_model(target_scale)

    def _compute_playback_total_time(self) -> None:
        bvh_time = self.animation.num_frames / self.animation.sample_rate if self.animation is not None else 0.0
        t2_time = self.t2_motion.duration if self.t2_motion is not None else 0.0
        wheel_time = self.wheel_motion.duration if self.wheel_motion is not None else 0.0
        self.playback_total_time = max(bvh_time, t2_time, wheel_time)
        self.playback_time = wp.clamp(self.playback_time, 0.0, self.playback_total_time)

    def _wheel_csv_for_t2_csv(self, t2_csv: Path) -> Path:
        self.generated_wheel_dir.mkdir(parents=True, exist_ok=True)
        return self.generated_wheel_dir / f"{t2_csv.stem}_diff_drive.csv"

    def _reload_t2_pair(self, t2_csv: Path, wheel_csv: Path | None = None) -> None:
        t2_csv = Path(t2_csv).expanduser().resolve()
        embedded_wheel_motion = _has_wheel_motion_columns(t2_csv)
        if wheel_csv is None:
            # A saved T3 CSV already contains the trajectory used by the viewer.
            # Reading it directly prevents a later load from deriving a different
            # path from the retargeted upper-body root.
            wheel_csv = t2_csv if embedded_wheel_motion else self._wheel_csv_for_t2_csv(t2_csv)
        else:
            wheel_csv = Path(wheel_csv).expanduser().resolve()
            if not wheel_csv.exists() and embedded_wheel_motion:
                wheel_csv = t2_csv
        should_generate_wheels = (
            self.base_mode in {"free", "wheels"}
            and wheel_csv != t2_csv
            and (self.args.regenerate_wheel_csv or not wheel_csv.exists())
        )
        if should_generate_wheels:
            _generate_wheel_csv(
                t2_csv,
                wheel_csv,
                self.args.fps,
                self.args.wheel_radius,
                self.args.waist_yaw_compensation if self.args.stiff_posture else 0.0,
            )

        self.current_t2_csv = t2_csv
        self.current_wheel_csv = wheel_csv
        self.t2_buffer = (
            csv_utils.load_csv(str(t2_csv), fps=self.args.fps, csv_config=csv_utils.get_csv_config("t2"))
            if _is_full_t2_csv(t2_csv)
            else None
        )
        self.t2_motion = _read_t2_motion(t2_csv, self.args.fps)
        if self.base_mode in {"free", "wheels"}:
            self.wheel_motion = _read_wheel_motion(wheel_csv, self.t2_motion.sample_rate, self.args.wheel_radius)
        elif self.base_mode == "fixed":
            self.wheel_motion = _fixed_wheel_motion(self.t2_motion.num_frames, self.t2_motion.sample_rate)
        else:
            self.wheel_motion = None
        self._compute_playback_total_time()

    def _human_hips_wheel_motion(self, sample_rate: float) -> WheelMotion | None:
        if self.skeleton is None or self.animation is None or self.skeleton_instance is None:
            return None

        base_joint_idx = self.skeleton.joint_index(self.args.base_joint)
        if base_joint_idx == -1:
            print(f"[WARN]: BVH does not expose base joint {self.args.base_joint!r}; cannot drive T3 base from human waist")
            return None

        # Use the same unscaled skeleton transforms that render the human. The
        # retargeting scaler is appropriate for IK effectors, but even a small
        # scale difference makes the two world trajectories slowly separate.
        base_targets = np.asarray(
            [
                self.animation.compute_global_transforms(frame_idx, self.skeleton_instance.xform)[base_joint_idx]
                for frame_idx in range(self.animation.num_frames)
            ],
            dtype=np.float64,
        )

        base_pos = base_targets[:, 0:3]
        root_xy_m = self.args.base_direction_sign * (base_pos[:, 0:2] - base_pos[0, 0:2])
        waist_heading = _joint_facing_heading(self.skeleton, base_targets, self.args.base_direction_sign)
        standing_motion = _is_standing_root_motion(root_xy_m, self.args.standing_motion_threshold)
        path_heading = _stable_path_headings(root_xy_m, sample_rate) if not standing_motion else waist_heading
        if not standing_motion:
            waist_heading = _align_heading_with_path(waist_heading, path_heading)
        else:
            # In-place clips have no movement direction to disambiguate the
            # generated BVH waist frame.  Apply a standing-only correction so
            # T3 faces the rendered human instead of the opposite side.
            waist_heading = np.unwrap(waist_heading + math.radians(self.args.standing_yaw_offset_deg))

        if self.args.base_yaw_source == "waist" or standing_motion:
            base_yaw = waist_heading
            drive_heading = waist_heading
            root_xy_m = _clamp_root_xy_radius(root_xy_m, self.args.standing_base_radius)
            if not standing_motion:
                root_xy_m = self.args.base_direction_sign * (base_pos[:, 0:2] - base_pos[0, 0:2])
        else:
            drive_heading = path_heading
            # Use the same absolute heading that is used to reconstruct the
            # non-holonomic wheel path, matching Kimodo's wheel-base playback.
            base_yaw = np.unwrap(drive_heading)

        return _wheel_motion_from_root_trajectory(
            root_xy_m=root_xy_m,
            yaw_rad=base_yaw,
            fps=sample_rate,
            wheel_radius_m=self.args.wheel_radius,
            wheel_separation_m=0.38,
            drive_heading_rad=drive_heading,
            enforce_no_slip=True,
        )

    def _robot_hand_local_xy(self, motion: T2Motion, hand_name: str) -> np.ndarray | None:
        body_name = "left_link7" if hand_name == "LeftHand" else "right_link7"
        body_names = [_name_from_label(label) for label in self.model.body_label]
        if body_name not in body_names:
            return None
        body_idx = body_names.index(body_name)

        saved_q = self.model.joint_q.numpy().copy()
        q = self.joint_q.copy()
        q[0:7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        local_xy = np.zeros((motion.num_frames, 2), dtype=np.float64)
        for frame_idx in range(motion.num_frames):
            for joint_name, values in motion.joint_angles.items():
                if self.args.stiff_posture and joint_name in T3_STIFF_POSTURE_JOINTS:
                    continue
                start = self.joint_q_start.get(joint_name)
                if start is not None:
                    q[start] = float(values[frame_idx])

            wp.copy(self.model.joint_q, wp.array(q, dtype=wp.float32), 0, 0, len(q))
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)
            body_q = self.state.body_q.numpy()
            local_xy[frame_idx] = np.asarray(body_q[body_idx][0:2], dtype=np.float64)

        wp.copy(self.model.joint_q, wp.array(saved_q, dtype=wp.float32), 0, 0, len(saved_q))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)
        return local_xy

    def _apply_hand_reach_base_correction(self, wheel_motion: WheelMotion, motion: T2Motion) -> WheelMotion:
        if self.args.base_reach_correction == "off":
            return wheel_motion
        if self.skeleton is None or self.animation is None or self.skeleton_instance is None:
            return wheel_motion

        scaler = HumanToRobotScaler(
            self.skeleton,
            1.8,
            io_utils.get_config_file("t2/soma_to_t2_scaler_config.json"))
        effector_names = scaler.effector_names()
        required = ["Hips", "LeftHand", "RightHand"]
        if any(name not in effector_names for name in required):
            return wheel_motion

        offset = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(*self.skeleton_instance.xform[3:7]))
        targets = scaler.compute_effectors_from_buffer(self.animation, True, offset)
        hips0 = np.asarray(targets[0, effector_names.index("Hips"), 0:2], dtype=np.float64)

        hand_names = ["LeftHand", "RightHand"]
        if self.args.base_reach_correction == "dominant":
            movement = []
            for hand_name in hand_names:
                hand_xy = np.asarray(targets[:, effector_names.index(hand_name), 0:2], dtype=np.float64)
                movement.append(float(np.max(np.linalg.norm(hand_xy - hand_xy[0], axis=1))))
            hand_names = [hand_names[int(np.argmax(movement))]]

        corrected_root_xy = np.stack([wheel_motion.x_m, wheel_motion.y_m], axis=1).astype(np.float64)
        yaw_rad = np.asarray(wheel_motion.yaw_rad, dtype=np.float64)
        correction_sum = np.zeros_like(corrected_root_xy)
        correction_count = 0

        for hand_name in hand_names:
            robot_local_xy = self._robot_hand_local_xy(motion, hand_name)
            if robot_local_xy is None:
                continue
            human_hand_xy = np.asarray(targets[:, effector_names.index(hand_name), 0:2], dtype=np.float64)
            human_hand_xy = self.args.base_direction_sign * (human_hand_xy - hips0)

            n = min(corrected_root_xy.shape[0], robot_local_xy.shape[0], human_hand_xy.shape[0])
            rotated_robot_xy = np.zeros((n, 2), dtype=np.float64)
            for frame_idx in range(n):
                c = math.cos(-yaw_rad[frame_idx])
                s = math.sin(-yaw_rad[frame_idx])
                rot = np.array([[c, -s], [s, c]], dtype=np.float64)
                rotated_robot_xy[frame_idx] = rot @ robot_local_xy[frame_idx]

            # Both arrays are already expressed in viewer-world metres. URDF
            # visual scale changes robot geometry, not world-space translation.
            desired_root_xy = human_hand_xy[:n] - rotated_robot_xy
            correction_sum[:n] += desired_root_xy - corrected_root_xy[:n]
            correction_count += 1

        if correction_count == 0:
            return wheel_motion

        standing_motion = _is_standing_root_motion(
            np.stack([wheel_motion.x_m, wheel_motion.y_m], axis=1),
            self.args.standing_motion_threshold,
        )
        correction = correction_sum / correction_count
        corrected_root_xy += float(self.args.base_reach_correction_gain) * correction
        if standing_motion:
            corrected_root_xy = _clamp_root_xy_radius(corrected_root_xy, self.args.standing_base_radius)
        return _wheel_motion_from_root_trajectory(
            root_xy_m=corrected_root_xy,
            yaw_rad=yaw_rad,
            fps=wheel_motion.sample_rate,
            wheel_radius_m=self.args.wheel_radius,
            wheel_separation_m=0.38,
        )

    def _retarget_soma_to_t3_motion(self) -> T2Motion | None:
        if self.skeleton is None or self.animation is None or self.skeleton_instance is None:
            return None

        scaler = HumanToRobotScaler(
            self.skeleton,
            1.8,
            io_utils.get_config_file("t2/soma_to_t2_scaler_config.json"))
        effector_names = scaler.effector_names()
        missing_effectors = [name for name in T3_IK_MAP if name not in effector_names]
        if missing_effectors:
            raise RuntimeError(f"T3 IK missing source effectors: {missing_effectors}")

        builder = newton.ModelBuilder()
        free_base = self.base_mode == "free"
        builder.add_urdf(str(self.args.t3_urdf), floating=False, scale=self.t3_scale)
        ik_model = builder.finalize(requires_grad=True)
        state = ik_model.state()
        newton.eval_fk(ik_model, ik_model.joint_q, ik_model.joint_qd, state)

        body_names = [_name_from_label(label) for label in ik_model.body_label]
        joint_q_start = _build_label_start_map(ik_model.joint_label, ik_model.joint_q_start.numpy())
        mapped = []
        for effector_name, (body_name, pos_weight, rot_weight) in T3_IK_MAP.items():
            if body_name not in body_names:
                raise RuntimeError(f"T3 IK body not found: {body_name}")
            mapped.append((effector_names.index(effector_name), body_names.index(body_name), pos_weight, rot_weight))

        offset = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(*self.skeleton_instance.xform[3:7]))
        targets = scaler.compute_effectors_from_buffer(self.animation, True, offset)
        hips_effector_idx = effector_names.index("Hips") if "Hips" in effector_names else None
        initial_hips_pos = (
            np.asarray(targets[0][hips_effector_idx][0:3], dtype=np.float64)
            if hips_effector_idx is not None
            else np.zeros(3, dtype=np.float64)
        )
        joint_q_np = np.expand_dims(ik_model.joint_q.numpy().copy(), axis=0).astype(np.float32)
        joint_q = wp.array(joint_q_np, dtype=wp.float32)

        position_objectives = []
        rotation_objectives = []
        for effector_idx, body_idx, pos_weight, rot_weight in mapped:
            initial = state.body_q.numpy()[body_idx]
            position_objectives.append(
                ik.IKObjectivePosition(
                    link_index=body_idx,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([wp.vec3(*initial[0:3])], dtype=wp.vec3),
                    weight=pos_weight))
            if rot_weight > 0.0:
                rotation_objectives.append(
                    ik.IKObjectiveRotation(
                        link_index=body_idx,
                        link_offset_rotation=wp.quat_identity(),
                        target_rotations=wp.array([wp.vec4(*initial[3:7])], dtype=wp.vec4),
                        weight=rot_weight))

        objectives = [
            *position_objectives,
            *rotation_objectives,
            ik.IKObjectiveJointLimit(
                joint_limit_lower=ik_model.joint_limit_lower,
                joint_limit_upper=ik_model.joint_limit_upper,
                weight=12.0),
        ]
        solver = ik.IKSolver(
            model=ik_model,
            n_problems=1,
            objectives=objectives,
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC)
        solver.reset()

        def single_step():
            solver.step(joint_q, joint_q, iterations=int(self.args.t3_ik_iterations))

        graph_capture = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                single_step()
            graph_capture = cap.graph
        else:
            single_step()

        joint_values: dict[str, list[float]] = {
            joint_name: [] for joint_name in T2_TO_T3_JOINTS.values()
        }
        root_pos_values: list[np.ndarray] = []
        root_quat_values: list[np.ndarray] = []
        print(
            f"[INFO]: Retargeting SOMA motion directly to {'free-base' if free_base else 'fixed-base'} T3 "
            f"({self.animation.num_frames} frames, {self.args.t3_ik_iterations} IK iterations/frame)")
        for frame_idx in range(self.animation.num_frames):
            if frame_idx > 0 and frame_idx % 120 == 0:
                print(f"[INFO]: T3 retarget progress: {frame_idx}/{self.animation.num_frames} frames")
            frame_targets = targets[frame_idx]
            root_delta = np.zeros(3, dtype=np.float64)
            root_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            if free_base and hips_effector_idx is not None:
                hips_target = np.asarray(frame_targets[hips_effector_idx], dtype=np.float64)
                root_delta = hips_target[0:3] - initial_hips_pos
                root_delta[2] = 0.0
                hips_euler = R.from_quat(hips_target[3:7]).as_euler("xyz", degrees=False)
                yaw_quat = _yaw_quat_z(float(hips_euler[2]))
                root_quat = np.array([yaw_quat[0], yaw_quat[1], yaw_quat[2], yaw_quat[3]], dtype=np.float64)
            rot_objective_idx = 0
            for objective_idx, (effector_idx, _, _, rot_weight) in enumerate(mapped):
                target = np.asarray(frame_targets[effector_idx], dtype=np.float64)
                target_pos = target[0:3] - root_delta
                position_objectives[objective_idx].set_target_position(0, wp.vec3(*target_pos))
                if rot_weight > 0.0:
                    rotation_objectives[rot_objective_idx].set_target_rotation(0, wp.vec4(*target[3:7]))
                    rot_objective_idx += 1

            if graph_capture is not None:
                wp.capture_launch(graph_capture)
            else:
                single_step()
            solved = joint_q.numpy()[0].copy()
            root_pos_values.append(root_delta.astype(np.float64))
            root_quat_values.append(root_quat.astype(np.float64))
            for joint_name in joint_values:
                start = joint_q_start.get(joint_name)
                joint_values[joint_name].append(float(solved[start]) if start is not None else 0.0)
        print("[INFO]: T3 retarget complete")

        return T2Motion(
            sample_rate=float(self.animation.sample_rate),
            joint_angles={name: np.asarray(values, dtype=np.float64) for name, values in joint_values.items()},
            root_pos=np.asarray(root_pos_values, dtype=np.float64),
            root_quat=np.asarray(root_quat_values, dtype=np.float64))

    def load_bvh_file(self, path: str | Path) -> None:
        path = Path(path).expanduser().resolve()
        if self.skeleton_renderer is not None and hasattr(self.viewer, "lines"):
            self.skeleton_renderer.clear(self.viewer)
        if self.skeletal_mesh_renderer is not None and hasattr(self.viewer, "objects"):
            self.skeletal_mesh_renderer.clear(self.viewer)
        if self.coordinate_renderer is not None and hasattr(self.viewer, "lines"):
            self.coordinate_renderer.clear(self.viewer)

        self.current_bvh = path
        self.skeleton, self.animation = bvh_utils.load_bvh(path)
        self.skeleton_instance = SkeletonInstance(
            self.skeleton,
            (235.0 / 255.0, 245.0 / 255.0, 112.0 / 255.0),
            self.converter.transform(wp.transform_identity()),
        )
        self.human_display_origin = self._compute_human_display_origin()
        self.skeleton_renderer = SkeletonRenderer(self.skeleton, [0])
        self.skeletal_mesh = pipeline_utils.get_source_model_mesh(pipeline_utils.SourceType.SOMA, self.skeleton)
        self.skeletal_mesh_renderer = SkeletalMeshRenderer(self.skeletal_mesh)
        self._compute_playback_total_time()

    def retarget_motion(self) -> None:
        if self.skeleton is None or self.animation is None or self.skeleton_instance is None:
            return

        if self.args.retarget_mode == "direct-t3":
            print("[WARN]: direct-t3 retarget is disabled because it does not preserve T3 arm pose quality. Using t2-copy.")

        import soma_retargeter.pipelines.newton_pipeline as newton_pipeline

        pipeline = newton_pipeline.NewtonPipeline(self.skeleton, "soma", "t2")
        r_offsets = [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(*self.skeleton_instance.xform[3:7]))]
        pipeline.add_input_motions([self.animation], r_offsets, True)
        buffers = pipeline.execute()
        if not buffers:
            return
        self.t2_buffer = buffers[0]
        motion_fps = float(self.t2_buffer.sample_rate)
        self.t2_motion = _t2_buffer_to_motion(self.t2_buffer, motion_fps)

        temp_dir = Path(tempfile.gettempdir()) / "soma-retargeter-t3-viewer"
        temp_dir.mkdir(parents=True, exist_ok=True)
        stem = self.current_bvh.stem if self.current_bvh is not None else "retargeted"
        t2_csv = temp_dir / f"{stem}_t3.csv"
        wheel_csv = temp_dir / f"{stem}_diff_drive.csv"
        if self.t2_buffer is not None:
            _save_t3_csv_from_buffer(t2_csv, self.t2_buffer)
        else:
            _save_t3_csv_from_motion(t2_csv, self.t2_motion)
        self.current_t2_csv = t2_csv
        self.current_wheel_csv = wheel_csv
        if self.base_mode in {"free", "wheels"}:
            self.wheel_motion = None
            if self.args.base_source == "human-hips":
                self.wheel_motion = self._human_hips_wheel_motion(motion_fps)
                if self.wheel_motion is not None:
                    self.wheel_motion = self._apply_hand_reach_base_correction(self.wheel_motion, self.t2_motion)
            if self.wheel_motion is None:
                _generate_wheel_csv(
                    t2_csv,
                    wheel_csv,
                    motion_fps,
                    self.args.wheel_radius,
                    self.args.waist_yaw_compensation if self.args.stiff_posture else 0.0,
                )
                self.wheel_motion = _read_wheel_motion(wheel_csv, motion_fps, self.args.wheel_radius)
            else:
                _save_wheel_motion_csv(wheel_csv, self.wheel_motion, self.args.wheel_radius)
        elif self.base_mode == "fixed":
            self.wheel_motion = _fixed_wheel_motion(self.t2_motion.num_frames, motion_fps)
        else:
            self.wheel_motion = None
        if self.t2_buffer is not None:
            _save_t3_csv_from_buffer(t2_csv, self.t2_buffer, self.wheel_motion, self.args.wheel_radius)
        else:
            _save_t3_csv_from_motion(t2_csv, self.t2_motion, self.wheel_motion, self.args.wheel_radius)
        self._compute_playback_total_time()

    def save_t2_csv(self, path: str | Path) -> None:
        if self.t2_buffer is None and self.t2_motion is None:
            return
        path = Path(path).expanduser().resolve()
        if self.t2_buffer is not None:
            _save_t3_csv_from_buffer(path, self.t2_buffer, self.wheel_motion, self.args.wheel_radius)
        else:
            _save_t3_csv_from_motion(path, self.t2_motion, self.wheel_motion, self.args.wheel_radius)
        wheel_csv = self._wheel_csv_for_t2_csv(path)
        if self.base_mode in {"free", "wheels"} and self.wheel_motion is not None:
            _save_wheel_motion_csv(wheel_csv, self.wheel_motion, self.args.wheel_radius)
        self.current_t2_csv = path
        self.current_wheel_csv = wheel_csv if self.base_mode in {"free", "wheels"} else None

    def _apply_t3_frame(self) -> None:
        q = self.joint_q.copy()
        t2_idx = self.t2_motion.frame_index(self.playback_time) if self.t2_motion is not None else 0

        if self.wheel_motion is not None:
            wheel_idx = self.wheel_motion.frame_index(self.playback_time)
            local_root_pos = wp.vec3(
                float(self.wheel_motion.x_m[wheel_idx]),
                float(self.wheel_motion.y_m[wheel_idx]),
                0.0,
            )
            # root_yaw_rad uses the mathematical ground-plane convention
            # atan2(+Y, +X), which is directly a +Z yaw in Newton.
            local_root_quat = _yaw_quat_z(float(self.wheel_motion.yaw_rad[wheel_idx]))
        elif self.base_mode == "free" and self.t2_motion is not None and self.t2_motion.root_pos is not None:
            local_root_pos = wp.vec3(*self.t2_motion.root_pos[t2_idx])
            root_quat = self.t2_motion.root_quat[t2_idx] if self.t2_motion.root_quat is not None else np.array([0.0, 0.0, 0.0, 1.0])
            local_root_quat = wp.quat(*root_quat)
        else:
            local_root_pos = wp.vec3(0.0, 0.0, 0.0)
            local_root_quat = wp.quat_identity()

        root_tx = wp.mul(self.t3_offset, wp.transform(local_root_pos, local_root_quat))
        self.t3_current_root_tx = wp.transform(root_tx)
        q[0:7] = _transform_to_numpy(root_tx)

        for joint_name, joint_value in T3_NEUTRAL_JOINTS.items():
            start = self.joint_q_start.get(joint_name)
            if start is not None:
                q[start] = joint_value

        if self.t2_motion is not None:
            for joint_name, values in self.t2_motion.joint_angles.items():
                if self.args.stiff_posture and joint_name in T3_STIFF_POSTURE_JOINTS:
                    continue
                start = self.joint_q_start.get(joint_name)
                if start is not None:
                    q[start] = float(values[t2_idx])

        left_start = self.joint_q_start.get("left_wheel_joint")
        right_start = self.joint_q_start.get("right_wheel_joint")
        if self.wheel_motion is not None and left_start is not None:
            wheel_idx = self.wheel_motion.frame_index(self.playback_time)
            # The viewer scales the complete URDF, including wheel radius. Use
            # the scaled visual radius here; exported wheel rates remain based
            # on the physical radius configured by --wheel-radius.
            visual_angle = float(self.wheel_motion.left_angle[wheel_idx]) / max(self.t3_scale, 1e-6)
            q[left_start] = math.remainder(visual_angle, 2.0 * math.pi)
        if self.wheel_motion is not None and right_start is not None:
            wheel_idx = self.wheel_motion.frame_index(self.playback_time)
            visual_angle = float(self.wheel_motion.right_angle[wheel_idx]) / max(self.t3_scale, 1e-6)
            q[right_start] = math.remainder(visual_angle, 2.0 * math.pi)

        wp.copy(self.model.joint_q, wp.array(q, dtype=wp.float32), 0, 0, len(q))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)

    def step(self):
        self.time += self.frame_dt
        if self.is_playing and self.playback_total_time > 0.0:
            self.playback_time += self.frame_dt * self.playback_speed
            if self.playback_loop:
                self.playback_time %= self.playback_total_time
            else:
                self.playback_time = max(0.0, min(self.playback_time, self.playback_total_time))

        self.human_offset = wp.transform(wp.vec3(self.human_offset.p[0], self.human_offset.p[1], 0.0), self.human_offset.q)
        self.t3_gizmo_offset = wp.transform(
            wp.vec3(self.t3_gizmo_offset.p[0], self.t3_gizmo_offset.p[1], self.t3_gizmo_offset.p[2]),
            self.t3_gizmo_offset.q,
        )
        self.t3_offset = self._gizmo_to_root_transform(self.t3_gizmo_offset)
        if self.skeleton_instance is not None and self.animation is not None:
            self.skeleton_instance.set_local_transforms(self.animation.sample(self.playback_time))
        self._apply_t3_frame()

    def render(self):
        self.viewer.begin_frame(self.time)
        if self.skeleton_instance is not None:
            prev_xform = wp.transform(self.skeleton_instance.xform)
            self.skeleton_instance.xform = wp.mul(
                self.human_offset,
                wp.mul(self.human_display_origin, self.skeleton_instance.xform),
            )
            if self.show_human_mesh and self.skeletal_mesh_renderer is not None:
                self.skeletal_mesh_renderer.draw(self.viewer, self.skeleton_instance, self.skeleton_instance.color, 0)
            if self.show_human_skeleton and self.skeleton_renderer is not None:
                self.skeleton_renderer.draw(self.viewer, self.skeleton_instance, 0)
            if self.show_joint_axes:
                self.coordinate_renderer.draw(self.viewer, self.skeleton_instance.compute_global_transforms(), 0.1, 0)
            self.skeleton_instance.xform = prev_xform

        if self.show_gizmos:
            self.viewer.log_gizmo("human_offset", self.human_offset)
            self.viewer.log_gizmo("robot_offset0", self.t3_gizmo_offset)
        self.viewer.log_state(self.state)
        if self.cpu_robot_mesh_renderer is not None:
            self.cpu_robot_mesh_renderer.draw(self.state)
        self.viewer.end_frame()

    def gui(self, ui):
        self.ui_playback_controls(ui)
        self.ui_scene_options(ui)

    def ui_scene_options(self, ui):
        import tkinter as tk
        from tkinter import filedialog as tk_filedialog

        viewport = ui.get_main_viewport()

        panel_size = ui.ImVec2(_UI_PANEL_WIDTH, 320)
        ui.set_next_window_pos(
            ui.ImVec2(
                viewport.size.x - _UI_PANEL_MARGIN - panel_size.x,
                viewport.size.y - _UI_PANEL_MARGIN - panel_size.y))
        ui.set_next_window_size(panel_size)
        ui.set_next_window_bg_alpha(_UI_PANEL_ALPHA)

        ui.begin("Scene Options", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        ui.separator()

        if ui.collapsing_header("Motion", flags=ui.TreeNodeFlags_.default_open):
            ui.separator()
            free_base = self.base_mode == "free"
            changed, free_base = ui.checkbox("Free T3 Base", free_base)
            if changed:
                self.base_mode = "free" if free_base else "fixed"
                if self.t2_motion is not None:
                    if self.base_mode == "free":
                        if self.current_t2_csv is not None:
                            self._reload_t2_pair(self.current_t2_csv, self.current_wheel_csv)
                    elif self.base_mode == "fixed":
                        self.wheel_motion = _fixed_wheel_motion(self.t2_motion.num_frames, self.t2_motion.sample_rate)
                        self.current_wheel_csv = None
                    elif self.current_t2_csv is not None:
                        self._reload_t2_pair(self.current_t2_csv, self.current_wheel_csv)
                self._compute_playback_total_time()
            ui.separator()
            ui.align_text_to_frame_padding()
            ui.text("BVH Motion:")
            ui.same_line()

            ui.push_id(100)
            if ui.button("Load"):
                root = tk.Tk()
                root.withdraw()
                bvh_path = tk_filedialog.askopenfilename(
                    title="Load BVH File",
                    defaultextension=".bvh",
                    filetypes=[("BVH files", "*.bvh")])
                if bvh_path:
                    self.load_bvh_file(bvh_path)
            ui.pop_id()

            if self.animation is None:
                ui.begin_disabled()
            ui.same_line()
            if ui.button("Retarget"):
                self.retarget_motion()
            if self.animation is None:
                ui.end_disabled()

            ui.align_text_to_frame_padding()
            ui.text("CSV Motion:")
            ui.same_line()

            ui.push_id(200)
            if ui.button("Load"):
                root = tk.Tk()
                root.withdraw()
                csv_path = tk_filedialog.askopenfilename(
                    title="Load T3 CSV File",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")])
                if csv_path:
                    self._reload_t2_pair(Path(csv_path))
            ui.pop_id()

            if self.t2_buffer is None and self.t2_motion is None:
                ui.begin_disabled()
            ui.same_line()
            if ui.button("Save"):
                root = tk.Tk()
                root.withdraw()
                save_path = tk_filedialog.asksaveasfilename(
                    title="Save T3 CSV File",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")])
                if save_path:
                    self.save_t2_csv(save_path)
            if self.t2_buffer is None and self.t2_motion is None:
                ui.end_disabled()

        ui.spacing()
        if ui.collapsing_header("Visibility", flags=ui.TreeNodeFlags_.default_open):
            ui.separator()

            changed, self.show_human_mesh = ui.checkbox("Show Mesh", self.show_human_mesh)
            if changed and self.skeletal_mesh_renderer is not None:
                self.skeletal_mesh_renderer.clear(self.viewer)
            changed, self.show_human_skeleton = ui.checkbox("Show Skeleton", self.show_human_skeleton)
            if changed and self.skeleton_renderer is not None:
                self.skeleton_renderer.clear(self.viewer)
            changed, self.show_joint_axes = ui.checkbox("Show Joint Axes", self.show_joint_axes)
            if changed and self.coordinate_renderer is not None:
                self.coordinate_renderer.clear(self.viewer)
            _, self.show_gizmos = ui.checkbox("Show Gizmos", self.show_gizmos)
            ui.same_line()
            if ui.button("Reset"):
                self.human_offset = wp.transform(self.default_human_offset)
                self.t3_offset = wp.transform(self.default_t3_offset)
                self.t3_gizmo_offset = wp.transform(self.default_t3_gizmo_offset)
        ui.end()

    def ui_playback_controls(self, ui):
        viewport = ui.get_main_viewport()

        panel_height = 105
        panel_width = viewport.size.x - 2 * (2 * _UI_PANEL_MARGIN + _UI_PANEL_WIDTH)

        ui.set_next_window_pos(
            ui.ImVec2(
                _UI_PANEL_WIDTH + _UI_PANEL_MARGIN,
                viewport.size.y - _UI_PANEL_MARGIN - panel_height))
        ui.set_next_window_size(ui.ImVec2(panel_width, panel_height))
        ui.set_next_window_bg_alpha(_UI_PANEL_ALPHA)

        ui.begin("Playback Controls", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        ui.align_text_to_frame_padding()
        ui.text("Time (s):")
        ui.same_line()
        ui.set_next_item_width(panel_width - 150)
        changed, new_time = ui.slider_float(
            "##TimeSlider",
            self.playback_time,
            0.0,
            self.playback_total_time,
            "%.2f")
        if changed:
            self.playback_time = wp.clamp(new_time, 0.0, self.playback_total_time)
        ui.same_line()
        ui.text_colored(ui.ImVec4(0.6, 0.8, 1.0, 1.0), f"{self.playback_total_time:.2f}s")

        self.is_playing = not ui.button("Pause") if self.is_playing else ui.button("Play ")
        ui.same_line()

        ui.align_text_to_frame_padding()
        ui.text("Speed")
        ui.same_line()
        ui.set_next_item_width(100)
        changed, new_speed = ui.slider_float(
            "##SpeedSlider",
            self.playback_speed,
            -2.0,
            2.0,
            "%.2f")
        if changed:
            self.playback_speed = new_speed
        ui.same_line()
        _, self.playback_loop = ui.checkbox("Loop", self.playback_loop)
        ui.end()

    def run(self):
        while self.viewer.is_running():
            self.step()
            self.render()
        self.viewer.close()


def parse_args():
    import argparse
    import newton.examples

    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="gl")
    parser.add_argument("--bvh", type=Path, default=None, help="Optional BVH to load at startup.")
    parser.add_argument("--t2-csv", type=Path, default=None, help="Optional T2/T3 CSV for T3 upper-body motion.")
    parser.add_argument("--wheel-csv", type=Path, default=None, help="Optional wheel/diff-drive CSV for T3 base motion.")
    parser.add_argument("--base-mode", choices=("free", "fixed", "wheels"), default="free", help="T3 base behavior. free uses wheel-base compensation; fixed locks base; wheels follows generated diff-drive motion.")
    parser.add_argument("--base-source", choices=("human-hips", "t2-root"), default="human-hips", help="For BVH retargeting, drive the T3 base from human hips/root motion or generated T2 root wheels.")
    parser.add_argument("--base-direction-sign", type=float, choices=(-1.0, 1.0), default=1.0, help="Human-hips trajectory direction. +1 keeps the T3 root in the same Newton world direction as the rendered SOMA human.")
    parser.add_argument("--base-joint", default="Hips", help="BVH joint used as the T3 base/waist source. Default: Hips.")
    parser.add_argument("--base-yaw-source", choices=("waist", "path"), default="waist", help="Use the BVH waist facing rotation or ground path tangent for T3 base yaw.")
    parser.add_argument("--standing-yaw-offset-deg", type=float, default=180.0, help="Extra yaw applied only to standing/in-place human-base clips.")
    parser.add_argument("--base-reach-correction", choices=("off", "dominant", "both"), default="off", help="Optionally shift the T3 base toward human hand targets. Off keeps the base exactly on the hips trajectory.")
    parser.add_argument("--base-reach-correction-gain", type=float, default=1.0, help="Gain for base reach correction. 1.0 fully applies the computed wrist-error correction.")
    parser.add_argument("--standing-motion-threshold", type=float, default=0.25, help="Treat clips with root displacement below this many meters as standing motions.")
    parser.add_argument("--standing-base-radius", type=float, default=0.04, help="Maximum XY base translation radius for standing motions, in meters.")
    parser.add_argument("--kimodo-root", type=Path, default=DEFAULT_KIMODO_ROOT, help="Kimodo checkout root for auto-generating wheel CSVs.")
    parser.add_argument("--t3-urdf", type=Path, default=DEFAULT_T3_URDF, help="T3 URDF path.")
    parser.add_argument("--fps", type=float, default=120.0, help="T2 and wheel CSV FPS.")
    parser.add_argument("--wheel-radius", type=float, default=0.10, help="Wheel radius used by generated wheel CSVs.")
    parser.add_argument("--waist-yaw-compensation", type=float, default=1.0, help="How much stiff T3 waist yaw is converted into wheel-base yaw.")
    parser.add_argument("--no-regenerate-wheel-csv", dest="regenerate_wheel_csv", action="store_false", help="Reuse existing wheel CSVs instead of regenerating them with current T3 settings.")
    parser.add_argument("--t3-scale", type=float, default=DEFAULT_T3_SCALE, help="T3 visual scale.")
    parser.add_argument("--retarget-mode", choices=("t2-copy", "direct-t3"), default="t2-copy", help="Retarget method. direct-t3 is currently disabled and falls back to t2-copy.")
    parser.add_argument("--t3-ik-iterations", type=int, default=12, help="IK iterations for direct SOMA-to-T3 retargeting.")
    parser.add_argument("--t3-x", type=float, default=0.0, help="T3 x offset.")
    parser.add_argument("--t3-y", type=float, default=0.0, help="T3 y offset.")
    parser.add_argument("--t3-z", type=float, default=0.0, help="T3 root z offset.")
    parser.add_argument("--robot-gizmo-pivot-x", type=float, default=0.0015099, help="Local T3 root X pivot used for the robot gizmo.")
    parser.add_argument("--robot-gizmo-pivot-y", type=float, default=0.0, help="Local T3 root Y pivot used for the robot gizmo.")
    parser.add_argument("--robot-gizmo-pivot-z", type=float, default=0.0, help="Local T3 root Z pivot used for the robot gizmo.")
    parser.add_argument("--human-x", type=float, default=0.0, help="Human x offset.")
    parser.add_argument("--human-y", type=float, default=0.0, help="Human y offset.")
    parser.add_argument("--follow-t2-posture", dest="stiff_posture", action="store_false", help="Let T3 copy T2 waist/head joints.")
    parser.set_defaults(stiff_posture=True)
    parser.set_defaults(regenerate_wheel_csv=True)
    return newton.examples.init(parser)


def main():
    viewer, args = parse_args()
    for path_name in ("t3_urdf",):
        path = getattr(args, path_name)
        if not path.exists():
            raise FileNotFoundError(f"{path_name.replace('_', '-')} not found: {path}")
    if args.bvh is not None and not args.bvh.exists():
        raise FileNotFoundError(f"bvh not found: {args.bvh}")
    if args.t2_csv is not None and not args.t2_csv.exists():
        raise FileNotFoundError(f"t2-csv not found: {args.t2_csv}")
    if args.wheel_csv is not None and not args.wheel_csv.exists():
        raise FileNotFoundError(f"wheel-csv not found: {args.wheel_csv}")
    if args.base_mode in {"free", "wheels"} and args.t2_csv is not None and args.wheel_csv is None:
        args.wheel_csv = REPO_ROOT / "assets" / "motions" / "t3" / "wheel_csv" / f"{args.t2_csv.stem}_diff_drive.csv"
    if (
        args.base_mode in {"free", "wheels"}
        and args.t2_csv is not None
        and (args.regenerate_wheel_csv or not args.wheel_csv.exists())
    ):
        print(f"[INFO]: Generating T3 wheel CSV from upper-body CSV: {args.wheel_csv}")
        _generate_wheel_csv(
            args.t2_csv,
            args.wheel_csv,
            args.fps,
            args.wheel_radius,
            args.waist_yaw_compensation if args.stiff_posture else 0.0,
        )
    with wp.ScopedDevice(args.device):
        app = T3HumanNewtonViewer(viewer, args)
        if args.bvh is not None:
            app.load_bvh_file(args.bvh)
        if args.t2_csv is not None:
            app._reload_t2_pair(args.t2_csv, args.wheel_csv)
        app.run()


if __name__ == "__main__":
    main()
