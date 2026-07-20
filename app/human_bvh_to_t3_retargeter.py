#!/usr/bin/env python3
"""Standalone human BVH -> T3 CSV retargeter.

This keeps the two parts of T3 separate, matching the Kimodo wheel-base tools:

1. Retarget the human upper-body BVH motion to T3/T2-compatible arm joints.
2. Derive the mobile base directly from the human BVH hips trajectory.
3. Save both a T3 CSV with embedded wheel columns and a separate diff-drive CSV.

The important bit is that the wheel base is not copied from the retargeted
humanoid pelvis after the fact.  It is generated as a real differential-drive
trajectory from the human ground path, so the wheels roll with the base motion.
"""

from __future__ import annotations

import argparse
import json
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
    DEFAULT_T3_EXPORT,
    DEFAULT_WHEEL_EXPORT,
    DEFAULT_WHEEL_RADIUS_M,
    DEFAULT_WHEEL_SEPARATION_M,
    T3_LIFT_HEIGHT_OFFSET_M,
    _append_wheel_columns_to_t3_csv,
    _append_lift_column_to_t3_csv,
    _convert_bvh_hips_to_wheels,
    _compute_bvh_lift_extensions,
    _save_t3_csv_from_t2_buffer,
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _collect_bvh_paths(args: argparse.Namespace, config: dict) -> list[Path]:
    paths: list[Path] = []
    if args.bvh is not None:
        paths.extend(args.bvh)

    if args.bvh_folder is not None:
        folder = args.bvh_folder.expanduser().resolve()
        if not folder.exists():
            raise FileNotFoundError(f"BVH folder not found: {folder}")
        paths.extend(sorted(folder.rglob("*.bvh")))

    if not paths and args.use_config_folder:
        import_folder = config.get("import_folder")
        if import_folder is None:
            raise ValueError("Config has no import_folder; pass --bvh or --bvh-folder")
        folder = (REPO_ROOT / import_folder).resolve()
        if not folder.exists():
            raise FileNotFoundError(f"BVH import folder not found: {folder}")
        paths.extend(sorted(folder.rglob("*.bvh")))

    resolved = []
    seen = set()
    for path in paths:
        bvh_path = path.expanduser().resolve()
        if bvh_path.suffix.lower() != ".bvh":
            continue
        if bvh_path in seen:
            continue
        if not bvh_path.exists():
            raise FileNotFoundError(f"BVH not found: {bvh_path}")
        seen.add(bvh_path)
        resolved.append(bvh_path)

    if not resolved:
        raise FileNotFoundError("No BVH files selected. Pass --bvh, --bvh-folder, or --use-config-folder.")
    return resolved


def _relative_output_path(bvh_path: Path, bvh_roots: list[Path], output_root: Path) -> Path:
    for root in bvh_roots:
        try:
            rel = bvh_path.relative_to(root)
            return (output_root / rel).with_suffix(".csv")
        except ValueError:
            pass
    return output_root / f"{bvh_path.stem}.csv"


def _retarget_one_bvh_to_t3_csv(
    bvh_path: Path,
    output_csv: Path,
    config: dict,
    lift_match_target: str = "waist",
    lift_height_offset_m: float = T3_LIFT_HEIGHT_OFFSET_M,
    include_lift_column: bool = True,
) -> float:
    import warp as wp

    import soma_retargeter.assets.bvh as bvh_utils
    import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str

    skeleton, animation = bvh_utils.load_bvh(bvh_path)
    converter = SpaceConverter(get_facing_direction_type_from_str(config.get("retarget_source_facing_direction", "Mujoco")))
    bvh_tx_converter = converter.transform(wp.transform_identity())

    pipeline = newton_pipeline.NewtonPipeline(
        skeleton,
        config.get("retarget_source", "soma"),
        "t2",
    )
    pipeline.clear()
    pipeline.add_input_motions([animation], [bvh_tx_converter], True)
    buffers = pipeline.execute()
    if len(buffers) != 1:
        raise RuntimeError(f"Expected one retargeted buffer for {bvh_path}, got {len(buffers)}")

    _save_t3_csv_from_t2_buffer(output_csv, buffers[0])
    if include_lift_column:
        lift_extensions = _compute_bvh_lift_extensions(
            bvh_path,
            config.get("retarget_source_facing_direction", "Mujoco"),
            lift_match_target,
            lift_height_offset_m,
        )
        _append_lift_column_to_t3_csv(output_csv, lift_extensions)
    return float(animation.sample_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget human BVH directly to T3 CSV. The T3 base follows the human BVH hips path "
            "as a differential-drive wheel trajectory."
        )
    )
    parser.add_argument("--bvh", type=Path, nargs="+", help="One or more BVH files to retarget.")
    parser.add_argument("--bvh-folder", type=Path, help="Folder of BVH files to retarget recursively.")
    parser.add_argument(
        "--use-config-folder",
        action="store_true",
        help="Retarget the config import_folder when --bvh/--bvh-folder is not provided.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Existing BVH converter config to reuse.")
    parser.add_argument("--t3-export-folder", type=Path, default=DEFAULT_T3_EXPORT, help="Output folder for T3 CSVs.")
    parser.add_argument("--wheel-export-folder", type=Path, default=DEFAULT_WHEEL_EXPORT, help="Output folder for wheel CSVs.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Wheel CSV FPS. Default: use each BVH animation sample_rate.",
    )
    parser.add_argument("--wheel-radius", type=float, default=DEFAULT_WHEEL_RADIUS_M, help="Wheel radius in meters.")
    parser.add_argument("--wheel-separation", type=float, default=DEFAULT_WHEEL_SEPARATION_M, help="Wheel separation in meters.")
    parser.add_argument("--max-forward-speed", type=float, default=3.0, help="Optional forward speed clamp in m/s.")
    parser.add_argument("--max-yaw-rate", type=float, default=12.0, help="Optional yaw-rate clamp in rad/s.")
    parser.add_argument(
        "--base-direction-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Human hips trajectory direction. +1 preserves the rendered SOMA human world direction.",
    )
    parser.add_argument(
        "--base-joint",
        default="Hips",
        help="BVH joint used as the T3 base/waist source. Default: Hips.",
    )
    parser.add_argument(
        "--base-yaw-source",
        choices=("waist", "path"),
        default="waist",
        help="Use the BVH waist facing rotation or ground path tangent for T3 base yaw. Default: waist.",
    )
    parser.add_argument(
        "--standing-yaw-offset-deg",
        type=float,
        default=180.0,
        help="Extra yaw applied only to standing/in-place human-base clips. Default: 180.",
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
        "--no-embed-wheel-columns",
        action="store_true",
        help="Do not append wheel/base columns into the T3 CSV.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = _load_json(config_path)

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

        print(f"[INFO]: [{idx}/{len(bvh_paths)}] Retargeting BVH to T3 upper body: {bvh_path}")
        sample_rate = _retarget_one_bvh_to_t3_csv(
            bvh_path,
            t3_csv,
            config,
            lift_match_target=args.lift_match_target,
            lift_height_offset_m=args.lift_height_offset_m,
            include_lift_column=not args.no_lift_column,
        )
        fps = float(args.fps) if args.fps is not None else sample_rate

        print(f"[INFO]: [{idx}/{len(bvh_paths)}] Generating T3 base from human {args.base_joint} path/facing: {wheel_csv}")
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

    print("[INFO]: Human BVH -> T3 retarget complete")
    print(f"[INFO]: T3 CSV folder: {t3_export}")
    print(f"[INFO]: Wheel CSV folder: {wheel_export}")
    for bvh_path, t3_csv, wheel_csv, fps in outputs:
        print(f"[INFO]: {bvh_path.name}")
        print(f"        T3 CSV:   {t3_csv}")
        print(f"        Wheel CSV:{wheel_csv}")
        print(f"        FPS:      {fps:g}")


if __name__ == "__main__":
    main()
