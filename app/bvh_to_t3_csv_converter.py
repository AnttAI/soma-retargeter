#!/usr/bin/env python3
"""All-in-one human BVH -> T3 CSV converter.

python ./app/bvh_to_t3_csv_converter.py --viewer gl

This file is the easy entry point for T3, similar to ``bvh_to_csv_converter.py``:

* ``--viewer gl`` keeps the existing Newton BVH/T3 visualizer behavior.
* ``--viewer null`` / ``--batch`` / ``--bvh`` runs the full BVH -> T3 CSV
  pipeline from the command line.

The conversion creates the T3 upper-body CSV, derives the two-wheel base from
the human BVH waist/hips trajectory, and embeds those wheel/base columns back
into the T3 CSV so Kimodo can play the whole robot in sync.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from bvh_to_t3_converter import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_KIMODO_ROOT,
    DEFAULT_T3_EXPORT,
    DEFAULT_WHEEL_EXPORT,
    DEFAULT_WHEEL_RADIUS_M,
    DEFAULT_WHEEL_SEPARATION_M,
    T3_LIFT_HEIGHT_OFFSET_M,
    _append_lift_column_to_t3_csv,
    _append_wheel_columns_to_t3_csv,
    _compute_bvh_lift_extensions,
    _convert_bvh_hips_to_wheels,
    _convert_t3_to_wheels,
    _load_json,
    _retarget_bvh_to_t3,
)
from human_bvh_to_t3_retargeter import (  # noqa: E402
    _collect_bvh_paths,
    _relative_output_path,
    _retarget_one_bvh_to_t3_csv,
)


_CONVERSION_FLAGS = {
    "--batch",
    "--headless",
    "--skip-retarget",
    "--use-config-folder",
    "--bvh",
    "--bvh-folder",
}


def _option_value(argv: list[str], option: str) -> str | None:
    prefix = f"{option}="
    for idx, token in enumerate(argv):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == option and idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def _wants_viewer(argv: list[str]) -> bool:
    if "--help" in argv or "-h" in argv:
        return False

    viewer = _option_value(argv, "--viewer")
    if viewer is not None:
        return viewer.lower() != "null"

    if any(flag in argv for flag in _CONVERSION_FLAGS):
        return False

    # Preserve the old behavior of this script: no args opens the T3 viewer.
    return True


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget human BVH to T3 CSV in one command. Use --viewer gl for the Newton viewer, "
            "or --viewer null / --batch for command-line conversion."
        )
    )
    parser.add_argument(
        "--viewer",
        choices=("gl", "null"),
        default="null",
        help="Use gl to open the Newton viewer. Use null for command-line conversion.",
    )
    parser.add_argument("--batch", "--headless", dest="batch", action="store_true", help="Run command-line conversion.")
    parser.add_argument("--bvh", type=Path, nargs="+", help="One or more BVH files to retarget.")
    parser.add_argument("--bvh-folder", type=Path, help="Folder of BVH files to retarget recursively.")
    parser.add_argument(
        "--use-config-folder",
        action="store_true",
        help="Retarget the config import_folder. If no --bvh/--bvh-folder is given, this is the batch default.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="BVH converter config to reuse.")
    parser.add_argument("--t3-export-folder", type=Path, default=DEFAULT_T3_EXPORT, help="Output folder for T3 CSVs.")
    parser.add_argument("--t2-export-folder", dest="t3_export_folder", type=Path, help="Deprecated alias for --t3-export-folder.")
    parser.add_argument("--wheel-export-folder", type=Path, default=DEFAULT_WHEEL_EXPORT, help="Output folder for wheel CSVs.")
    parser.add_argument("--kimodo-root", type=Path, default=DEFAULT_KIMODO_ROOT, help="Kimodo checkout root for the viewer hint.")
    parser.add_argument("--fps", type=float, default=None, help="Wheel CSV FPS. Default: BVH sample rate for selected BVHs, 30 for config batch.")
    parser.add_argument("--wheel-radius", type=float, default=DEFAULT_WHEEL_RADIUS_M, help="Wheel radius in meters.")
    parser.add_argument("--wheel-separation", type=float, default=DEFAULT_WHEEL_SEPARATION_M, help="Wheel separation in meters.")
    parser.add_argument("--max-forward-speed", type=float, default=3.0, help="Optional forward speed clamp in m/s.")
    parser.add_argument("--max-yaw-rate", type=float, default=12.0, help="Optional yaw-rate clamp in rad/s.")
    parser.add_argument("--yaw-source", choices=("auto", "path", "t2"), default="auto", help="Fallback wheel yaw source when base-source=t2-root.")
    parser.add_argument("--waist-yaw-compensation", type=float, default=1.0, help="Fallback T2-root yaw compensation. Use 0 to disable.")
    parser.add_argument(
        "--base-source",
        choices=("human-hips", "t2-root"),
        default="human-hips",
        help="Generate T3 base motion from the matching human BVH hips or from retargeted T2 root.",
    )
    parser.add_argument(
        "--base-direction-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Human hips trajectory direction. +1 preserves the rendered SOMA human world direction.",
    )
    parser.add_argument("--base-joint", default="Hips", help="BVH joint used as the T3 base/waist source.")
    parser.add_argument(
        "--base-yaw-source",
        choices=("waist", "path"),
        default="waist",
        help="Use the BVH waist facing rotation or ground path tangent for T3 base yaw.",
    )
    parser.add_argument(
        "--standing-yaw-offset-deg",
        type=float,
        default=180.0,
        help="Extra yaw applied only to standing/in-place clips.",
    )
    parser.add_argument(
        "--standing-motion-threshold",
        type=float,
        default=0.25,
        help="Treat clips with root displacement below this many meters as standing/in-place motions.",
    )
    parser.add_argument(
        "--standing-base-radius",
        type=float,
        default=0.04,
        help="Maximum XY base translation radius for standing/in-place motions, in meters.",
    )
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
    parser.add_argument("--skip-retarget", action="store_true", help="Only regenerate wheel/base columns for existing T3 CSVs.")
    parser.add_argument("--no-embed-wheel-columns", action="store_true", help="Save separate wheel CSVs but do not embed wheel columns into T3 CSVs.")
    return parser.parse_args(argv)


def _run_selected_bvhs(args: argparse.Namespace, config: dict) -> list[tuple[Path, Path, Path, float]]:
    bvh_paths = _collect_bvh_paths(args, config)
    t3_export = args.t3_export_folder.expanduser().resolve()
    wheel_export = args.wheel_export_folder.expanduser().resolve()

    bvh_roots: list[Path] = []
    if args.bvh_folder is not None:
        bvh_roots.append(args.bvh_folder.expanduser().resolve())
    if config.get("import_folder"):
        bvh_roots.append((REPO_ROOT / config["import_folder"]).resolve())

    source_facing_direction = config.get("retarget_source_facing_direction", "Mujoco")
    outputs: list[tuple[Path, Path, Path, float]] = []

    for idx, bvh_path in enumerate(bvh_paths, start=1):
        t3_csv = _relative_output_path(bvh_path, bvh_roots, t3_export)
        wheel_csv = _relative_output_path(bvh_path, bvh_roots, wheel_export).with_name(f"{bvh_path.stem}_diff_drive.csv")

        if args.skip_retarget:
            if not t3_csv.exists():
                raise FileNotFoundError(f"--skip-retarget requested, but T3 CSV does not exist: {t3_csv}")
            sample_rate = 30.0
            print(f"[INFO]: [{idx}/{len(bvh_paths)}] Reusing existing T3 CSV: {t3_csv}")
        else:
            print(f"[INFO]: [{idx}/{len(bvh_paths)}] Retargeting BVH to T3 upper body: {bvh_path}")
            sample_rate = _retarget_one_bvh_to_t3_csv(
                bvh_path,
                t3_csv,
                config,
                lift_match_target=args.lift_match_target,
                lift_height_offset_m=args.lift_height_offset_m,
                include_lift_column=not args.no_lift_column,
            )

        if args.skip_retarget and not args.no_lift_column:
            lift_extensions = _compute_bvh_lift_extensions(
                bvh_path,
                source_facing_direction,
                args.lift_match_target,
                args.lift_height_offset_m,
            )
            _append_lift_column_to_t3_csv(t3_csv, lift_extensions)

        fps = float(args.fps) if args.fps is not None else float(sample_rate)
        print(f"[INFO]: [{idx}/{len(bvh_paths)}] Generating T3 base/wheels from human {args.base_joint}: {wheel_csv}")
        _convert_bvh_hips_to_wheels(
            bvh_path=bvh_path,
            wheel_csv=wheel_csv,
            fps=fps,
            wheel_radius_m=args.wheel_radius,
            wheel_separation_m=args.wheel_separation,
            max_forward_speed=args.max_forward_speed,
            max_yaw_rate=args.max_yaw_rate,
            source_facing_direction=source_facing_direction,
            base_direction_sign=args.base_direction_sign,
            standing_motion_threshold=args.standing_motion_threshold,
            standing_base_radius=args.standing_base_radius,
            base_joint_name=args.base_joint,
            base_yaw_source=args.base_yaw_source,
            standing_yaw_offset_deg=args.standing_yaw_offset_deg,
        )

        if not args.no_embed_wheel_columns:
            _append_wheel_columns_to_t3_csv(t3_csv, wheel_csv, fps)

        outputs.append((bvh_path, t3_csv, wheel_csv, fps))

    return outputs


def _run_config_batch(args: argparse.Namespace, config: dict) -> list[tuple[Path, Path, Path, float]]:
    t3_export = args.t3_export_folder.expanduser().resolve()
    wheel_export = args.wheel_export_folder.expanduser().resolve()
    fps = float(args.fps) if args.fps is not None else 30.0
    bvh_import_root = (REPO_ROOT / config["import_folder"]).resolve() if "import_folder" in config else None
    source_facing_direction = config.get("retarget_source_facing_direction", "Mujoco")

    if not args.skip_retarget:
        print("[INFO]: Retargeting config import_folder BVHs to T3 CSVs")
        _retarget_bvh_to_t3(
            config,
            t3_export,
            lift_match_target=args.lift_match_target,
            lift_height_offset_m=args.lift_height_offset_m,
            include_lift_column=not args.no_lift_column,
        )

    print("[INFO]: Generating T3 base/wheel columns")
    generated = _convert_t3_to_wheels(
        t3_export=t3_export,
        wheel_export=wheel_export,
        bvh_import_root=bvh_import_root,
        fps=fps,
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
    return [(Path(), t3_csv, wheel_csv, fps) for t3_csv, wheel_csv in generated]


def _print_summary(args: argparse.Namespace, outputs: list[tuple[Path, Path, Path, float]]) -> None:
    t3_export = args.t3_export_folder.expanduser().resolve()
    wheel_export = args.wheel_export_folder.expanduser().resolve()
    kimodo_root = args.kimodo_root.expanduser().resolve()
    t3_viewer = kimodo_root / "wheel_base_tools" / "view_t3_robot.py"
    t3_urdf = kimodo_root / "robot_demo_outputs" / "t3_robot" / "T3.urdf"

    print("[INFO]: Human BVH -> T3 CSV complete")
    print(f"[INFO]: T3 CSV folder: {t3_export}")
    print(f"[INFO]: Wheel CSV folder: {wheel_export}")
    print(f"[INFO]: Motions processed: {len(outputs)}")
    for bvh_path, t3_csv, wheel_csv, fps in outputs[:10]:
        label = bvh_path.name if str(bvh_path) else t3_csv.name
        print(f"[INFO]: {label}")
        print(f"        T3 CSV:    {t3_csv}")
        print(f"        Wheel CSV: {wheel_csv}")
        print(f"        FPS:       {fps:g}")
    if len(outputs) > 10:
        print(f"[INFO]: ... {len(outputs) - 10} more")

    if t3_viewer.exists():
        fps = outputs[0][3] if outputs else (float(args.fps) if args.fps is not None else 30.0)
        print("[INFO]: Optional stiff T3 viewer command:")
        print(
            "  "
            f"{sys.executable} {t3_viewer} "
            f"--t3-urdf {t3_urdf} "
            f"--t2-csv-root {t3_export} "
            f"--wheel-csv-root {wheel_export} "
            f"--fps {fps:g}"
        )


def _run_converter(argv: list[str]) -> None:
    args = _parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = _load_json(config_path)

    if args.bvh is not None or args.bvh_folder is not None:
        outputs = _run_selected_bvhs(args, config)
    else:
        outputs = _run_config_batch(args, config)

    _print_summary(args, outputs)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if _wants_viewer(argv):
        from t3_human_newton_viewer import main as viewer_main

        viewer_main()
        return

    _run_converter(argv)


if __name__ == "__main__":
    main()
