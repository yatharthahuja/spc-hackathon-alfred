"""Replay a sequence of saved poses by their YAML keys.

Loads ``arm_poses.yaml`` (or ``--file``), takes a list of pose names from
the command line, lifts the arm to the auto-captured home pose (= enable
pose + 0.2 m in Z, same as ``trajectory.py``), then visits each requested
name in order by driving directly to its saved 6-joint configuration with
``Robot.move_to_joints`` (pure joint-space interpolation, no IK).

Usage
-----
    # Replay three named poses in order:
    python replay_trajectory.py home pickup drop

    # Use a different file:
    python replay_trajectory.py --file my_poses.yaml home pickup

    # See what's available:
    python replay_trajectory.py --list

    # Skip the initial home lift and/or the final zero-return:
    python replay_trajectory.py --no-home --no-zero-return pickup drop

The YAML format matches what ``keyboard_teleop.py`` writes:

    home_pose:
      x: ...   y: ...   z: ...
      roll: ...   pitch: ...   yaw: ...
      gripper: ...
      joints: [j1, j2, j3, j4, j5, j6]
      timestamp: '...'

Only the ``joints`` and ``gripper`` fields are used here (xyz/rpy are
ignored — we replay in joint space to avoid IK ambiguity).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

from trajectory import Robot, Tolerances


DEFAULT_YAML = Path("arm_poses.yaml")
N_JOINTS = 6


# ──────────────────────────────────────────────────────────────────────
# YAML helpers
# ──────────────────────────────────────────────────────────────────────

def load_poses(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"poses file not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"top-level YAML in {path} must be a mapping")
    return data


def validate_entry(name: str, entry: object) -> Tuple[np.ndarray, float]:
    """Return ``(joints[6], gripper)`` for ``entry`` or raise ValueError."""
    if not isinstance(entry, dict):
        raise ValueError(
            f"entry {name!r} must be a mapping, got {type(entry).__name__}"
        )
    joints = entry.get("joints")
    if joints is None:
        raise ValueError(f"entry {name!r} has no 'joints' field")
    arr = np.asarray(joints, dtype=np.float64).reshape(-1)
    if arr.shape != (N_JOINTS,):
        raise ValueError(
            f"entry {name!r} 'joints' must be {N_JOINTS} floats, got shape {arr.shape}"
        )
    gripper = float(entry.get("gripper", 0.0))
    return arr, gripper


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "names", nargs="*",
        help="Pose keys to visit, in order.",
    )
    parser.add_argument(
        "--file", "-f", type=Path, default=DEFAULT_YAML,
        help=f"YAML file with saved poses (default: {DEFAULT_YAML}).",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List available pose names and exit.",
    )
    parser.add_argument(
        "--segment-duration", "-d", type=float, default=3.0,
        help="Seconds per segment between waypoints (default 3.0).",
    )
    parser.add_argument(
        "--home-duration", type=float, default=3.0,
        help="Seconds for the initial lift to home (default 3.0).",
    )
    parser.add_argument(
        "--zero-duration", type=float, default=3.0,
        help="Seconds for the final return to all-zero joints (default 3.0).",
    )
    parser.add_argument(
        "--no-home", action="store_true",
        help="Skip the initial move to home pose.",
    )
    parser.add_argument(
        "--no-zero-return", action="store_true",
        help="Skip the final return to all-zero joints.",
    )
    parser.add_argument(
        "--joint-tol", type=float, default=0.03,
        help="Per-joint settle tolerance in radians (default 0.03).",
    )
    parser.add_argument(
        "--gripper-tol", type=float, default=0.1,
        help="Gripper settle tolerance in radians (default 0.10).",
    )
    parser.add_argument(
        "--continue-on-failure", action="store_true",
        help="Don't abort the sequence if a waypoint times out.",
    )
    parser.add_argument(
        "--interpolate-gripper", action="store_true",
        help="Linearly interpolate the gripper during the joint motion "
             "(default: hold gripper until joints reach the target).",
    )
    return parser.parse_args()


def _list_poses(poses: dict, path: Path) -> None:
    if not poses:
        print(f"(no poses saved in {path})")
        return
    print(f"Poses in {path}:")
    name_w = max((len(n) for n in poses), default=4)
    for n, e in poses.items():
        if not isinstance(e, dict):
            print(f"  {n:<{name_w}}   (malformed entry)")
            continue
        joints = e.get("joints") or []
        grip = float(e.get("gripper", 0.0))
        ts = e.get("timestamp", "")
        joints_str = ", ".join(f"{float(v):+.3f}" for v in joints)
        print(f"  {n:<{name_w}}   joints=[{joints_str}]   grip={grip:+.3f}   {ts}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    try:
        poses = load_poses(args.file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.list:
        _list_poses(poses, args.file)
        return 0

    if not args.names:
        print(
            "Error: no pose names given. "
            f"Use --list to see what's available in {args.file}.",
            file=sys.stderr,
        )
        args.names = ["pre_pick_marker", "pick_marker", "post_pick_marker", "pre_place", "drop_marker"]

        # args.names = ["pre_pick_marker", "pick_marker", "post_pick_marker"]

    missing = [n for n in args.names if n not in poses]
    if missing:
        print(f"Error: unknown pose names: {missing}", file=sys.stderr)
        print(f"Available: {list(poses.keys())}", file=sys.stderr)
        return 2

    entries: List[Tuple[str, np.ndarray, float]] = []
    for n in args.names:
        try:
            joints, grip = validate_entry(n, poses[n])
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 3
        entries.append((n, joints, grip))

    print(f"=== Replay {len(entries)} pose(s) from {args.file} ===")
    for n, j, g in entries:
        joints_str = ", ".join(f"{float(v):+.3f}" for v in j)
        print(f"  {n}: joints=[{joints_str}]   grip={g:+.3f}")
    print()

    tol = Tolerances(joint=args.joint_tol, gripper=args.gripper_tol)

    success = True
    with Robot() as r:
        print(f"Start pose: {r.get_pose()}")
        print(f"Home pose:  {r.home_pose}  (start + (0, 0, +0.2 m))")

        if not args.no_home:
            print("→ home")
            if not r.home(duration=args.home_duration):
                print("WARNING: home() did not settle within tolerance.")

        gripper_after = not args.interpolate_gripper
        for i, (name, joints, grip) in enumerate(entries, 1):
            print(f"[{i}/{len(entries)}] → {name}")
            ok = r.move_to_joints(
                joints,
                gripper=grip,
                duration=args.segment_duration,
                wait=True,
                tolerances=tol,
                gripper_after=gripper_after,
            )
            cur_q = r.get_joint_positions()
            max_err = float(np.max(np.abs(cur_q - joints)))
            print(
                f"    {'reached' if ok else 'NOT REACHED'} "
                f"(max joint err {max_err:.4f} rad)"
            )
            if not ok and not args.continue_on_failure:
                print(f"Aborting replay at {name!r}.")
                success = False
                break

        if not args.no_zero_return:
            print("→ returning to all-zero joints")
            r.move_to_joints(
                np.zeros(N_JOINTS),
                gripper=0.0,
                duration=args.zero_duration,
                wait=True,
                tolerances=Tolerances(
                    joint=max(tol.joint, 0.05),
                    gripper=max(tol.gripper, 0.1),
                ),
            )

    print("Replay complete." if success else "Replay aborted.")
    return 0 if success else 4


if __name__ == "__main__":
    sys.exit(main())
