"""Programmatic robot driver: execute one or more named poses on demand.

Same data source and same hardware path as ``replay_trajectory.py``
(``arm_poses.yaml`` + ``trajectory.Robot.move_to_joints`` in joint space),
but exposed as a small Python API so other code — e.g. a voice / agent
loop in ``interactive/elevenlabs.py`` — can drive the arm with a single
function call.

Quick examples
--------------
    from robot import execute, Sequencer

    # 1) One-shot, single pose (opens, lifts to home, moves, returns
    #    to zero, disconnects):
    execute("home_pose")

    # 2) One-shot sequence (same as replay_trajectory.py:186-187):
    execute([
        "pre_pick_marker", "pick_marker", "post_pick_marker",
        "pre_place", "drop_marker",
    ])

    # 3) Multi-step session with ONE hardware connection across many calls
    #    (best for interactive use; avoids re-enabling motors each time):
    with Sequencer() as s:
        s.execute("home_pose")
        s.execute(["pre_pick_marker", "pick_marker"])
        s.execute("drop_marker")

CLI
---
    python robot.py home_pose pre_pick_marker pick_marker drop_marker
    python robot.py                       # runs DEFAULT_SEQUENCE
    python robot.py --list                # show available pose names

Defaults
--------
- Home-first on connect (Z + 0.2 m above the enable pose, set by
  ``trajectory.Robot``).
- Zero-joint return on disconnect.
- Gripper is held during joint motion and only actuates after the joints
  arrive at each waypoint (``gripper_after=True`` in ``move_to_joints``).
"""

from __future__ import annotations

import argparse
import atexit
import signal
import sys
import threading
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import yaml

# Allow ``python robot.py`` from anywhere by anchoring imports + the default
# YAML path to this file's directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from trajectory import Robot, Tolerances  # noqa: E402


DEFAULT_YAML: Path = _HERE / "arm_poses.yaml"
N_JOINTS: int = 6

# Fallback used by the CLI when no names are given (matches
# replay_trajectory.py:186-187 — handy default for a pick-and-place demo).
DEFAULT_SEQUENCE: List[str] = [
    "pre_pick_marker",
    "pick_marker",
    "post_pick_marker",
    "pre_place",
    "drop_marker",
]


# ──────────────────────────────────────────────────────────────────────
# YAML helpers
# ──────────────────────────────────────────────────────────────────────

def load_poses(path: Union[str, Path] = DEFAULT_YAML) -> dict:
    """Read the named-poses YAML and return its top-level mapping."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"poses file not found: {p}")
    with p.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"top-level YAML in {p} must be a mapping")
    return data


def _extract(name: str, entry: object) -> Tuple[np.ndarray, float]:
    """Return ``(joints[6], gripper)`` for ``entry`` or raise ``ValueError``."""
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
            f"entry {name!r} 'joints' must be {N_JOINTS} floats, "
            f"got shape {arr.shape}"
        )
    return arr, float(entry.get("gripper", 0.0))


def _normalise_names(names: Union[str, Iterable[str]]) -> List[str]:
    """Accept a single name OR an iterable of names; return a list."""
    if isinstance(names, str):
        return [names]
    try:
        out = list(names)
    except TypeError:
        raise TypeError(
            "names must be a string or an iterable of strings, "
            f"got {type(names).__name__}"
        )
    if not all(isinstance(n, str) for n in out):
        raise TypeError("every entry in names must be a string")
    return out


# ──────────────────────────────────────────────────────────────────────
# Sequencer — keeps one open hardware connection across many execute() calls
# ──────────────────────────────────────────────────────────────────────

class Sequencer:
    """Holds an open ``Robot`` connection and runs named pose sequences.

    Construct once, then call ``execute(...)`` as many times as needed.
    Lifts to home on the first ``connect()``. On ``disconnect()`` (or at
    the end of a ``with`` block) optionally returns to all-zero joints and
    then disables the motors in place — same lifecycle as
    ``replay_trajectory.py``.

    Parameters
    ----------
    poses_file : str | Path
        YAML to load (default: ``arm_poses.yaml`` next to this file).
    segment_duration : float
        Seconds per waypoint motion (default 3.0).
    home_duration, zero_duration : float
        Seconds for the initial home lift and final zero return (default 3.0).
    home_on_connect : bool
        If True (default), lift the arm to its captured home pose on first
        ``connect()``.
    zero_on_disconnect : bool
        If True (default), return to all-zero joints before disabling.
    joint_tol, gripper_tol : float
        Acceptance tolerances passed to ``Robot.move_to_joints``.
    interpolate_gripper : bool
        If True, interpolate the gripper alongside the joint motion. If
        False (default), hold the gripper until the joints arrive (then
        actuate). Matches ``replay_trajectory.py``'s default.
    continue_on_failure : bool
        If True, a missed waypoint doesn't abort the rest of the sequence.
    verbose : bool
        Print progress lines (default True).
    install_safety_handlers : bool
        If True (default), install an ``atexit`` hook plus SIGTERM/SIGHUP
        handlers that drive the arm to all-zero joints and disable the
        motors if the program crashes or is killed cleanly. SIGKILL,
        segfaults, and sudden power loss cannot be intercepted — for those,
        toggle the 24 V supply to reset.
    emergency_duration : float
        Seconds for the emergency zero-return motion (default 2.0). Kept
        short so a crashing process doesn't hang the shell.
    emergency_timeout : float
        Hard timeout in seconds for the emergency zero-return; the disable
        runs even if zero-return overruns (default 4.0).
    """

    # Class-level lock prevents two emergency cleanups (e.g. atexit + signal
    # arriving back-to-back) from racing on the same hardware.
    _shutdown_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        poses_file: Union[str, Path] = DEFAULT_YAML,
        *,
        segment_duration: float = 3.0,
        home_duration: float = 3.0,
        zero_duration: float = 3.0,
        home_on_connect: bool = True,
        zero_on_disconnect: bool = True,
        joint_tol: float = 0.03,
        gripper_tol: float = 0.1,
        interpolate_gripper: bool = False,
        continue_on_failure: bool = False,
        verbose: bool = True,
        install_safety_handlers: bool = True,
        emergency_duration: float = 2.0,
        emergency_timeout: float = 4.0,
    ) -> None:
        self.poses_file = Path(poses_file)
        self.poses = load_poses(self.poses_file)
        self.segment_duration = float(segment_duration)
        self.home_duration = float(home_duration)
        self.zero_duration = float(zero_duration)
        self.home_on_connect = bool(home_on_connect)
        self.zero_on_disconnect = bool(zero_on_disconnect)
        self.tol = Tolerances(joint=float(joint_tol), gripper=float(gripper_tol))
        self.gripper_after = not bool(interpolate_gripper)
        self.continue_on_failure = bool(continue_on_failure)
        self.verbose = bool(verbose)
        self.install_safety_handlers = bool(install_safety_handlers)
        self.emergency_duration = float(emergency_duration)
        self.emergency_timeout = float(emergency_timeout)

        self._robot: Optional[Robot] = None
        self._homed: bool = False
        self._shutdown_done: bool = False

        # Safety-handler bookkeeping (only set while a connection is open).
        self._atexit_fn: Optional[Callable[[], None]] = None
        self._prev_signal_handlers: Dict[int, object] = {}

    # ── connection lifecycle ──────────────────────────────────────────

    def connect(self) -> "Sequencer":
        """Open the hardware connection and lift to home (if configured)."""
        if self._robot is None:
            self._log("Connecting to robot ...")
            self._robot = Robot()
            self._robot.connect()
            self._homed = False
            self._shutdown_done = False
            if self.install_safety_handlers:
                self._install_handlers()
            self._log(f"Start pose: {self._robot.get_pose()}")
            self._log(f"Home pose:  {self._robot.home_pose}")
        if self.home_on_connect and not self._homed:
            self._log("→ home")
            if not self._robot.home(duration=self.home_duration):
                self._log("WARNING: home() did not settle within tolerance.")
            self._homed = True
        return self

    def disconnect(self) -> None:
        """Optionally return to all-zero joints, then disable + close the bus.

        This is the clean shutdown path (used by ``__exit__``). See
        :meth:`safe_shutdown` for the panic / crash path.
        """
        if self._robot is None:
            return
        try:
            if self.zero_on_disconnect:
                self._log("→ returning to all-zero joints")
                self._robot.move_to_joints(
                    np.zeros(N_JOINTS),
                    gripper=0.0,
                    duration=self.zero_duration,
                    wait=True,
                    tolerances=Tolerances(
                        joint=max(self.tol.joint, 0.05),
                        gripper=max(self.tol.gripper, 0.1),
                    ),
                )
        finally:
            self._robot.disconnect()
            self._robot = None
            self._homed = False
            self._shutdown_done = True
            self._remove_handlers()

    def __enter__(self) -> "Sequencer":
        return self.connect()

    def __exit__(self, *args) -> None:
        self.disconnect()

    # ── safety net: emergency shutdown ────────────────────────────────

    def safe_shutdown(self, *, reason: str = "") -> None:
        """Drive the arm to all-zero joints and disable the motors NOW.

        Safe to call from anywhere — context manager exit, exception
        handler, atexit hook, signal handler, another thread. Idempotent:
        repeat calls are no-ops once shutdown has happened.

        The zero-return runs with a short hard timeout
        (``emergency_timeout``); the disable runs **regardless** of
        whether the zero-return succeeded, so a stuck arm still ends up
        de-energised at the end.
        """
        with self._shutdown_lock:
            if self._shutdown_done or self._robot is None:
                return
            self._shutdown_done = True  # claim it now so reentrant calls bail
            tag = f" ({reason})" if reason else ""
            self._log(f"!! safe_shutdown{tag}: zeroing joints + disabling motors")
            try:
                self._robot.move_to_joints(
                    np.zeros(N_JOINTS),
                    gripper=0.0,
                    duration=self.emergency_duration,
                    wait=True,
                    timeout=self.emergency_timeout,
                    tolerances=Tolerances(joint=0.1, gripper=0.2),
                )
            except Exception as e:
                print(
                    f"[robot] safe_shutdown: zero-return failed: {e}",
                    file=sys.stderr,
                )
            try:
                self._robot.disconnect()
            except Exception as e:
                print(
                    f"[robot] safe_shutdown: disconnect failed: {e}",
                    file=sys.stderr,
                )
            self._robot = None
            self._homed = False
            self._remove_handlers()

    # ── handler install / remove ──────────────────────────────────────

    def _install_handlers(self) -> None:
        # atexit covers normal exit, sys.exit(), and unhandled exceptions.
        if self._atexit_fn is None:
            self._atexit_fn = lambda: self.safe_shutdown(reason="atexit")
            atexit.register(self._atexit_fn)

        # Signal handlers can only be installed from the main thread.
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGTERM, signal.SIGHUP):
            try:
                prev = signal.signal(sig, self._on_signal)
                # Remember previous handler so we can restore on disconnect.
                self._prev_signal_handlers[sig] = prev
            except (OSError, ValueError, AttributeError):
                # Some platforms (e.g. Windows) lack SIGHUP; ignore.
                pass

    def _remove_handlers(self) -> None:
        if self._atexit_fn is not None:
            try:
                atexit.unregister(self._atexit_fn)
            except Exception:
                pass
            self._atexit_fn = None
        for sig, prev in self._prev_signal_handlers.items():
            try:
                signal.signal(sig, prev)
            except (OSError, ValueError):
                pass
        self._prev_signal_handlers.clear()

    def _on_signal(self, signum: int, _frame) -> None:
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = str(signum)
        try:
            self.safe_shutdown(reason=f"signal {name}")
        except Exception:
            traceback.print_exc()
        # Exit with the conventional 128+signum so callers can tell why.
        sys.exit(128 + int(signum))

    # ── core API ──────────────────────────────────────────────────────

    def execute(self, names: Union[str, Iterable[str]]) -> bool:
        """Drive the arm through the requested pose name(s) in order.

        ``names`` may be a single string or any iterable of strings.
        Returns ``True`` iff every waypoint is reached within tolerance.
        Lazily connects (and lifts to home on first connect) if you
        haven't already.
        """
        names_list = _normalise_names(names)
        if not names_list:
            self._log("execute(): no names given.")
            return True

        missing = [n for n in names_list if n not in self.poses]
        if missing:
            raise KeyError(
                f"unknown pose name(s): {missing}. "
                f"Available: {list(self.poses)}"
            )

        # Resolve all entries up-front so a malformed YAML fails fast.
        entries = [(n, *_extract(n, self.poses[n])) for n in names_list]

        if self._robot is None:
            self.connect()
        robot = self._robot
        assert robot is not None  # for type-checkers

        success = True
        for i, (name, joints, grip) in enumerate(entries, 1):
            self._log(f"[{i}/{len(entries)}] → {name}")
            try:
                ok = robot.move_to_joints(
                    joints,
                    gripper=grip,
                    duration=self.segment_duration,
                    wait=True,
                    tolerances=self.tol,
                    gripper_after=self.gripper_after,
                )
            except BaseException as e:
                # Anything from a CallError to a KeyboardInterrupt: get the
                # arm to a safe state before bubbling the exception up.
                print(
                    f"[robot] move_to_joints raised during {name!r}: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                try:
                    self.safe_shutdown(reason=f"exception in {name!r}")
                except Exception:
                    traceback.print_exc()
                raise
            cur_q = robot.get_joint_positions()
            max_err = float(np.max(np.abs(cur_q - joints)))
            self._log(
                f"    {'reached' if ok else 'NOT REACHED'} "
                f"(max joint err {max_err:.4f} rad)"
            )
            if not ok and not self.continue_on_failure:
                self._log(f"Aborting at {name!r}.")
                success = False
                break
        return success

    # ── convenience accessors ─────────────────────────────────────────

    def list_poses(self) -> List[str]:
        """Names of all poses currently loaded from the YAML."""
        return list(self.poses.keys())

    def has(self, name: str) -> bool:
        return name in self.poses

    def reload(self) -> None:
        """Re-read the YAML in case it changed (e.g. teleop saved a new pose)."""
        self.poses = load_poses(self.poses_file)

    def get_pose(self):
        """Current end-effector ``Pose`` (xyz+rpy+gripper)."""
        if self._robot is None:
            self.connect()
        return self._robot.get_pose()  # type: ignore[union-attr]

    def get_joint_positions(self) -> np.ndarray:
        if self._robot is None:
            self.connect()
        return self._robot.get_joint_positions()  # type: ignore[union-attr]

    # ── internals ─────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ──────────────────────────────────────────────────────────────────────
# One-shot convenience: execute() opens + runs + closes
# ──────────────────────────────────────────────────────────────────────

def execute(
    names: Union[str, Iterable[str]],
    *,
    poses_file: Union[str, Path] = DEFAULT_YAML,
    **kwargs,
) -> bool:
    """Connect, run the named pose(s), and disconnect — all in one call.

    Equivalent to::

        with Sequencer(poses_file, **kwargs) as s:
            return s.execute(names)
    """
    with Sequencer(poses_file=poses_file, **kwargs) as s:
        return s.execute(names)


# ──────────────────────────────────────────────────────────────────────
# CLI (same shape as replay_trajectory.py, but uses Sequencer underneath)
# ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "names", nargs="*",
        help=f"Pose names to visit, in order. If omitted, runs the built-in "
             f"DEFAULT_SEQUENCE: {DEFAULT_SEQUENCE}.",
    )
    p.add_argument(
        "--file", "-f", type=Path, default=DEFAULT_YAML,
        help=f"YAML file with saved poses (default: {DEFAULT_YAML.name}).",
    )
    p.add_argument("--list", "-l", action="store_true",
                   help="List available pose names and exit.")
    p.add_argument("--segment-duration", "-d", type=float, default=3.0)
    p.add_argument("--home-duration", type=float, default=3.0)
    p.add_argument("--zero-duration", type=float, default=3.0)
    p.add_argument("--no-home", action="store_true")
    p.add_argument("--no-zero-return", action="store_true")
    p.add_argument("--joint-tol", type=float, default=0.03)
    p.add_argument("--gripper-tol", type=float, default=0.1)
    p.add_argument("--continue-on-failure", action="store_true")
    p.add_argument("--interpolate-gripper", action="store_true",
                   help="Interpolate gripper during motion (default: hold "
                        "until joints arrive, then actuate).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        poses = load_poses(args.file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.list:
        if not poses:
            print(f"(no poses saved in {args.file})")
        else:
            for n in poses:
                print(n)
        return 0

    names: List[str] = list(args.names) if args.names else list(DEFAULT_SEQUENCE)

    # Validate names against the already-loaded YAML *before* connecting,
    # so a typo doesn't trigger a full enable/home/disable cycle.
    missing = [n for n in names if n not in poses]
    if missing:
        print(f"Error: unknown pose names: {missing}", file=sys.stderr)
        print(f"Available: {list(poses.keys())}", file=sys.stderr)
        return 2

    seq = Sequencer(
        poses_file=args.file,
        segment_duration=args.segment_duration,
        home_duration=args.home_duration,
        zero_duration=args.zero_duration,
        home_on_connect=not args.no_home,
        zero_on_disconnect=not args.no_zero_return,
        joint_tol=args.joint_tol,
        gripper_tol=args.gripper_tol,
        interpolate_gripper=args.interpolate_gripper,
        continue_on_failure=args.continue_on_failure,
    )

    try:
        with seq:
            ok = seq.execute(names)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    print("Sequence complete." if ok else "Sequence aborted.")
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
