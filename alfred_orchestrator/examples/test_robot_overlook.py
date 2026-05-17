from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_DRIVERS_DIR = PROJECT_ROOT / "robot_drivers"
DEFAULT_POSES_FILE = ROBOT_DRIVERS_DIR / "arm_poses.yaml"

if str(ROBOT_DRIVERS_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOT_DRIVERS_DIR))

from robot import Sequencer, load_poses  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move the robot to the overlook pose, then return to base. "
            "By default, base is the driver's all-zero joint return on disconnect."
        )
    )
    parser.add_argument("--poses-file", type=Path, default=DEFAULT_POSES_FILE)
    parser.add_argument("--overlook-pose", default="overlook")
    parser.add_argument(
        "--base-pose",
        help=(
            "Optional named pose to visit after overlook. If omitted, the script "
            "returns to all-zero joints during disconnect."
        ),
    )
    parser.add_argument("--segment-duration", type=float, default=3.0)
    parser.add_argument("--home-duration", type=float, default=3.0)
    parser.add_argument("--zero-duration", type=float, default=3.0)
    parser.add_argument(
        "--no-home",
        action="store_true",
        help="Skip the driver's home lift before moving to overlook.",
    )
    parser.add_argument(
        "--no-zero-return",
        action="store_true",
        help="Do not return to all-zero joints on disconnect.",
    )
    parser.add_argument(
        "--list-poses",
        action="store_true",
        help="Print available pose names and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    poses = load_poses(args.poses_file)

    if args.list_poses:
        print("Available poses:")
        for pose_name in poses:
            print(f"  {pose_name}")
        return 0

    required_poses = [args.overlook_pose]
    if args.base_pose:
        required_poses.append(args.base_pose)

    missing = [pose_name for pose_name in required_poses if pose_name not in poses]
    if missing:
        print(f"Missing pose(s) in {args.poses_file}: {missing}", file=sys.stderr)
        print(f"Available poses: {list(poses)}", file=sys.stderr)
        return 2

    print(f"Using poses file: {args.poses_file}")
    print(f"Overlook pose: {args.overlook_pose}")
    if args.base_pose:
        print(f"Return pose: {args.base_pose}")
    elif args.no_zero_return:
        print("Return pose: none (--no-zero-return set)")
    else:
        print("Return pose: all-zero joints on disconnect")

    sequencer = Sequencer(
        poses_file=args.poses_file,
        segment_duration=args.segment_duration,
        home_duration=args.home_duration,
        zero_duration=args.zero_duration,
        home_on_connect=not args.no_home,
        zero_on_disconnect=not args.no_zero_return,
        verbose=True,
    )

    try:
        print("Connecting to robot...")
        sequencer.connect()

        print(f"Moving to {args.overlook_pose!r}...")
        if not sequencer.execute(args.overlook_pose):
            print(f"Robot did not settle at {args.overlook_pose!r}.", file=sys.stderr)
            return 4

        if args.base_pose:
            print(f"Moving to return pose {args.base_pose!r}...")
            if not sequencer.execute(args.base_pose):
                print(f"Robot did not settle at {args.base_pose!r}.", file=sys.stderr)
                return 5

        print("Robot overlook test succeeded.")
        return 0
    except Exception as exc:
        print(f"Robot overlook test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        print("Disconnecting robot; zero-return runs here unless disabled.")
        sequencer.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
