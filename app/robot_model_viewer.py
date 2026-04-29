# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import newton
import warp as wp

try:
    from viewer_compat import enable_cpu_pinned_fallback
except ModuleNotFoundError:
    from app.viewer_compat import enable_cpu_pinned_fallback

try:
    from cpu_robot_mesh_renderer import CpuRobotMeshRenderer
except ModuleNotFoundError:
    from app.cpu_robot_mesh_renderer import CpuRobotMeshRenderer


_REPO_ROOT = Path(__file__).resolve().parent.parent
_TARA_DIR = _REPO_ROOT / "tara"
_T2_DIR = _REPO_ROOT / "antt_t2"

_ROBOT_MODEL_PATHS = {
    "tara": _TARA_DIR / "T1_serial.xml",
    "t2": _T2_DIR / "T2_serial_nero_arms.urdf",
}
_ROBOT_MODEL_SCALES = {
    # Matches Tara's visual height to the rendered SOMA human mesh height at frame 0.
    "tara": 1.4121780259421193,
    # Matches T2's visual height to the rendered SOMA human mesh height at frame 0.
    "t2": 0.9589869491995866,
}
_ROBOT_MODEL_SPAWN_OFFSETS = {
    # Source Tara MJCF keeps the feet 1.005646 m above its root frame.
    # We drop by that amount (then apply the uniform robot scale) so the soles land on z=0.
    "tara": (0.0, 0.0, -1.0056460005066394),
    "t2": (0.0, 0.0, 1.032823),
}
_ROBOT_MODEL_JOINT_Q_OVERRIDES = {}


def _resolve_robot_asset(robot: str, mjcf_path: str | None) -> Path:
    if mjcf_path is not None:
        path = Path(mjcf_path).expanduser().resolve()
    elif robot == "unitree_g1":
        return newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml"
    else:
        path = _ROBOT_MODEL_PATHS.get(robot)
        if path is None:
            allowed = ", ".join(sorted([*list(_ROBOT_MODEL_PATHS.keys()), "unitree_g1"]))
            raise ValueError(f"Unknown robot [{robot}]. Allowed values: {allowed}")

    if not path.exists():
        raise FileNotFoundError(f"Robot asset file not found: {path}")

    return path


def _get_robot_scale(robot_name: str | None) -> float:
    if robot_name is None:
        return 1.0
    return _ROBOT_MODEL_SCALES.get(robot_name, 1.0)


def _apply_robot_joint_q_overrides(model, robot_name: str | None) -> None:
    if robot_name is None:
        return

    joint_overrides = _ROBOT_MODEL_JOINT_Q_OVERRIDES.get(robot_name)
    if not joint_overrides:
        return

    joint_q_values = model.joint_q.numpy()
    joint_q_starts = model.joint_q_start.numpy()

    for joint_label, joint_q_start in zip(model.joint_label, joint_q_starts):
        joint_name = joint_label.rsplit("/", 1)[-1]
        if joint_name in joint_overrides:
            joint_q_values[int(joint_q_start)] = joint_overrides[joint_name]

    wp.copy(model.joint_q, wp.array(joint_q_values, dtype=wp.float32), 0, 0, len(joint_q_values))


def _add_robot_asset(builder: newton.ModelBuilder, robot_asset: Path, robot_scale: float) -> None:
    if robot_asset.suffix.lower() == ".urdf":
        builder.add_urdf(str(robot_asset), floating=True, scale=robot_scale)
    else:
        builder.add_mjcf(str(robot_asset), scale=robot_scale)


def _build_model(robot_asset: Path, num_robots: int):
    robot_name = next((name for name, path in _ROBOT_MODEL_PATHS.items() if path == robot_asset), None)
    robot_scale = _get_robot_scale(robot_name)

    robot_builder = newton.ModelBuilder()
    _add_robot_asset(robot_builder, robot_asset, robot_scale)

    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    base_offset = _ROBOT_MODEL_SPAWN_OFFSETS.get(robot_name, (0.0, 0.0, 0.0))

    for i in range(num_robots):
        offset = wp.transform(
            wp.vec3(
                base_offset[0] * robot_scale,
                base_offset[1] * robot_scale + i - (num_robots - 1) / 2.0,
                base_offset[2] * robot_scale,
            ),
            wp.quat_identity(),
        )
        builder.add_builder(robot_builder, offset)

    model = builder.finalize()
    _apply_robot_joint_q_overrides(model, robot_name)
    return model


def main():
    import newton.examples

    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="gl")
    parser.add_argument(
        "--robot",
        type=str,
        default="t2",
        help="Robot asset to load. Supported values: t2, tara, unitree_g1.",
    )
    parser.add_argument(
        "--mjcf",
        type=lambda x: None if x == "None" else str(x),
        default=None,
        help="Optional explicit MJCF or URDF path. Overrides --robot when set.",
    )
    parser.add_argument(
        "--num-robots",
        type=int,
        default=1,
        help="Number of robot instances to spawn in the scene.",
    )

    viewer, args = newton.examples.init(parser)
    if args.num_robots < 1:
        raise ValueError("--num-robots must be >= 1")

    robot_asset = _resolve_robot_asset(args.robot, args.mjcf)

    with wp.ScopedDevice(args.device):
        model = _build_model(robot_asset, args.num_robots)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state, None)

        title = args.robot if args.mjcf is None else robot_asset.stem
        if hasattr(viewer, "renderer"):
            viewer.renderer.set_title(f"Robot Model Viewer - {title}")
        enable_cpu_pinned_fallback(viewer)
        viewer.set_model(model)
        viewer.set_world_offsets([0, 0, 0])
        cpu_robot_mesh_renderer = None
        if isinstance(viewer, newton.viewer.ViewerGL) and not viewer.device.is_cuda:
            cpu_robot_mesh_renderer = CpuRobotMeshRenderer(viewer, model)

        time = 0.0
        frame_dt = 1.0 / 60.0
        while viewer.is_running():
            time += frame_dt
            viewer.begin_frame(time)
            viewer.log_state(state)
            if cpu_robot_mesh_renderer is not None:
                cpu_robot_mesh_renderer.draw(state)
            viewer.end_frame()

        viewer.close()


if __name__ == "__main__":
    main()
