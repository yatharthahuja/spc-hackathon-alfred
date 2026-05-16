"""Keyboard teleoperation for the reBot Arm B601-DM.

Drives the arm by sending xyz + rpy + gripper commands from keypresses,
through the :class:`Robot` API in ``trajectory.py``.

Keys (single press; hold to auto-repeat in most terminals)
----------------------------------------------------------
Translation (robot base frame: +X forward, +Y left, +Z up):
    w / s   +X / -X
    a / d   +Y / -Y
    r / f   +Z / -Z
Rotation (XYZ Euler, radians):
    u / o   +roll  / -roll
    i / k   +pitch / -pitch
    j / l   +yaw   / -yaw
Gripper (continuous: each press is one step):
    n       open   (-gripper)
    m       close  (+gripper)
    [ / ]   aliases for n / m
    ; / '   decrease / increase the gripper step
    G       prompt for an absolute gripper value (radians)
System:
    h       go to home pose (= enable pose + 0.2 m in Z)
    0       go to all-zero joints (then resync target)
    p       print current pose and target
    S       save current pose + joints to a YAML file (prompts for a name).
            Appends to the existing file if present; if a pose with the
            same name already exists it is overwritten in place.
    L       load a saved pose from the YAML file (prompts for a name).
            Drives the arm to the saved joint configuration, then sets
            the gripper to the saved value (arm first, gripper second).
    SPACE   halt motion at current pose (resyncs target to where the arm is)
    +/-     increase / decrease translation step
    < / >   increase / decrease rotation step   (',' and '.' keys)
    ?       show help
    ESC / q quit. Forces a move to all-zero joints first if the arm is not
            already there, then disables motors and closes.

Setup
-----
1. ``/dev/ttyACM0`` must be readable+writable (``sudo chmod 666 /dev/ttyACM0``).
2. No motorbridge-gateway should be running (it would hold the serial port).
3. Run from the same directory as ``trajectory.py``:
       python keyboard_teleop.py
"""

from __future__ import annotations

import math
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import yaml

from trajectory import Pose, Robot, Tolerances


# ──────────────────────────────────────────────────────────────────────
# Tunables
# ──────────────────────────────────────────────────────────────────────

TRANSLATION_STEP_INIT = 0.01      # m
ROTATION_STEP_INIT = 0.05         # rad (~2.9 deg)
GRIPPER_STEP_INIT = 0.05          # rad — gripper increment per n/m press

MOVE_DURATION = 0.3               # s — short so teleop feels responsive
MOVE_TIMEOUT = 1.0                # s
HOME_OFFSET = (0.0, 0.0, 0.2)     # +0.2 m above the enable pose

POSES_YAML = "arm_poses.yaml"     # where 'S' saves named poses (cwd by default)
ZERO_JOINT_TOL = 0.03             # rad — below this, "q" skips the zero move

# Soft Cartesian workspace bounds. IK will reject unreachable poses anyway, but
# these guard against accidentally driving way out and getting stuck against
# joint limits. Adjust to taste.
X_RANGE = (0.05, 0.65)
Y_RANGE = (-0.40, 0.40)
Z_RANGE = (0.0, 0.55)
ROLL_RANGE = (-math.pi, math.pi)
PITCH_RANGE = (-math.pi, math.pi)
YAW_RANGE = (-math.pi, math.pi)
GRIPPER_RANGE = (-3.5, 0.0)

HELP = """\
Translation:  w/s = +-x   a/d = +-y   r/f = +-z
Rotation:     u/o = +-roll   i/k = +-pitch   j/l = +-yaw
Gripper:      n = open   m = close   ([ / ] are aliases)
              G = set absolute gripper value (prompts)
              ; / ' = decrease / increase gripper step
System:       h = home   0 = zero joints   p = print pose
              S = save current pose+joints (prompts for name; overwrites
                  if the name already exists)
              L = load saved pose (prompts for name)
              SPACE = halt   ? = help
              +/- = trans step   < / > = rot step
              ESC or q = quit (zeros joints first, then disables)
"""


# ──────────────────────────────────────────────────────────────────────
# Raw-mode keyboard helpers
# ──────────────────────────────────────────────────────────────────────

class RawStdin:
    """Context manager: put stdin into cbreak mode for single-key reads.

    Also exposes :meth:`cooked` for temporarily restoring line-buffered mode
    so we can call ``input()`` (e.g. to prompt for a pose name).
    """

    def __enter__(self) -> "RawStdin":
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *args) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    @contextmanager
    def cooked(self) -> Iterator[None]:
        """Temporarily restore cooked (line-buffered) mode."""
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        try:
            yield
        finally:
            tty.setcbreak(self.fd)


def _read_key(timeout: float) -> Optional[str]:
    """Return one keystroke (or ESC sequence) within ``timeout`` seconds."""
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        # ESC alone or the start of an arrow / function key sequence
        r2, _, _ = select.select([sys.stdin], [], [], 0.005)
        if r2:
            ch += sys.stdin.read(2)
    return ch


def _say(msg: str = "") -> None:
    """Print a line in raw mode (carriage return + newline)."""
    sys.stdout.write(msg + "\r\n")
    sys.stdout.flush()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _clamp_pose(p: Pose) -> Pose:
    return Pose(
        x=_clamp(p.x, *X_RANGE),
        y=_clamp(p.y, *Y_RANGE),
        z=_clamp(p.z, *Z_RANGE),
        roll=_clamp(p.roll, *ROLL_RANGE),
        pitch=_clamp(p.pitch, *PITCH_RANGE),
        yaw=_clamp(p.yaw, *YAW_RANGE),
        gripper=_clamp(p.gripper, *GRIPPER_RANGE),
    )


def _save_pose_entry(
    name: str,
    pose: Pose,
    joints: np.ndarray,
    *,
    path: str = POSES_YAML,
) -> tuple[bool, bool]:
    """Append ``name`` to the poses YAML file.

    Loads ``path`` if it exists, updates / inserts the ``name`` key, and
    writes back. Returns ``(success, existed)`` — ``existed`` is True if
    the key was already present and got overwritten.
    """
    p = Path(path)
    data: dict = {}
    if p.exists():
        try:
            with p.open("r") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                data = loaded
            elif loaded is None:
                data = {}
            else:
                print(
                    f"[teleop] {p} is not a YAML mapping; refusing to overwrite",
                    file=sys.stderr,
                )
                return False, False
        except Exception as e:
            print(f"[teleop] could not parse {p}: {e}", file=sys.stderr)
            return False, False

    existed = name in data
    data[name] = {
        "x": round(float(pose.x), 6),
        "y": round(float(pose.y), 6),
        "z": round(float(pose.z), 6),
        "roll": round(float(pose.roll), 6),
        "pitch": round(float(pose.pitch), 6),
        "yaw": round(float(pose.yaw), 6),
        "gripper": round(float(pose.gripper), 6),
        "joints": [round(float(v), 6) for v in joints],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        with p.open("w") as f:
            yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)
    except Exception as e:
        print(f"[teleop] failed to write {p}: {e}", file=sys.stderr)
        return False, existed
    return True, existed


def _load_poses(path: str = POSES_YAML) -> dict:
    """Read the named-poses YAML; returns ``{}`` if absent or malformed."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"[teleop] could not parse {p}: {e}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _extract_joints_and_gripper(
    name: str, entry: object,
) -> Optional[tuple[np.ndarray, float]]:
    """Pull ``(joints[6], gripper)`` from a saved YAML entry, or None on error."""
    if not isinstance(entry, dict):
        return None
    joints = entry.get("joints")
    if joints is None:
        return None
    try:
        arr = np.asarray(joints, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.shape != (6,):
        return None
    grip = float(entry.get("gripper", 0.0))
    return arr, grip


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== reBot Arm B601-DM keyboard teleop ===")
    print("Connecting to /dev/ttyACM0 ...")

    trans_step = TRANSLATION_STEP_INIT
    rot_step = ROTATION_STEP_INIT
    grip_step = GRIPPER_STEP_INIT

    with Robot(home_offset=HOME_OFFSET) as r:
        start = r.get_pose()
        home = r.home_pose
        print(f"Start pose: {start}")
        print(f"Home pose:  {home}  (start + {HOME_OFFSET})")
        print("Lifting to home ...")
        if not r.home(duration=3.0):
            print("WARNING: home() did not settle within tolerance — continuing.")
        target = r.get_pose()
        print(f"At home:    {target}")
        print(HELP)

        last_status = ""
        with RawStdin() as raw:
            while True:
                key = _read_key(timeout=0.05)
                if key is None:
                    continue

                dirty = False  # set True when target changes and we need a new move

                # ── quit ──────────────────────────────────────────────
                if key in ("\x1b", "q", "Q"):
                    _say("")
                    q_now = r.get_joint_positions()
                    max_dev = float(np.max(np.abs(q_now)))
                    if max_dev > ZERO_JOINT_TOL:
                        _say(f"Quit: forcing move to all-zero joints "
                             f"(max joint dev {max_dev:.3f} rad)...")
                        r.move_to_joints(
                            np.zeros(6), gripper=0.0,
                            duration=3.0, wait=True,
                        )
                    else:
                        _say(f"Quit: already at zero joints "
                             f"(max dev {max_dev:.3f} rad).")
                    _say("Disabling motors ...")
                    break

                # ── help / state ──────────────────────────────────────
                elif key in ("?",):
                    for line in HELP.splitlines():
                        _say(line)
                elif key == "p":
                    cur = r.get_pose()
                    _say(f"current : {cur}")
                    _say(f"target  : {target}")
                    _say(f"step    : trans={trans_step*1000:.0f} mm  "
                         f"rot={math.degrees(rot_step):.1f} deg  "
                         f"grip={grip_step:.3f} rad")

                # ── save current pose + joints to YAML ────────────────
                elif key == "S":
                    cur_pose = r.get_pose()
                    cur_joints = r.get_joint_positions()
                    _say("")
                    _say("Save current pose+joints to YAML.")
                    _say(f"  current: {cur_pose}")
                    _say(f"  joints : {[round(float(v), 4) for v in cur_joints]}")
                    try:
                        with raw.cooked():
                            name = input(f"Enter name for this pose "
                                         f"(blank to cancel) > ").strip()
                    except (EOFError, KeyboardInterrupt):
                        name = ""
                    if not name:
                        _say("Save cancelled.")
                    else:
                        ok, existed = _save_pose_entry(
                            name, cur_pose, cur_joints, path=POSES_YAML,
                        )
                        if ok:
                            note = " (overwritten)" if existed else ""
                            _say(f"Saved '{name}'{note} → {POSES_YAML}")
                        else:
                            _say(f"Save FAILED for '{name}'.")

                # ── load a saved pose from YAML and move there ────────
                elif key == "L":
                    saved = _load_poses(POSES_YAML)
                    if not saved:
                        _say(f"No poses available in {POSES_YAML}.")
                        continue
                    _say("")
                    _say(f"Saved poses in {POSES_YAML}:")
                    for n in saved:
                        _say(f"  - {n}")
                    try:
                        with raw.cooked():
                            name = input(
                                "Enter pose name to load (blank to cancel) > "
                            ).strip()
                    except (EOFError, KeyboardInterrupt):
                        name = ""
                    if not name:
                        _say("Load cancelled.")
                        continue
                    if name not in saved:
                        _say(f"No pose named {name!r}. Cancelled.")
                        continue
                    extracted = _extract_joints_and_gripper(name, saved[name])
                    if extracted is None:
                        _say(f"Pose {name!r} has no valid 'joints' field.")
                        continue
                    joints, grip = extracted
                    _say(
                        f"→ load {name!r}: joints="
                        f"{[round(float(v),3) for v in joints]}  "
                        f"grip={grip:+.3f}"
                    )
                    ok = r.move_to_joints(
                        joints,
                        gripper=grip,
                        duration=3.0,
                        wait=True,
                        gripper_after=True,
                    )
                    if ok:
                        _say(f"Reached {name!r}.")
                    else:
                        _say(f"Did not fully settle at {name!r} "
                             f"(see above for details).")
                    # Resync the teleop target so further w/a/s/d/... keys are
                    # relative to where the arm actually is now.
                    target = r.get_pose()

                # ── system motions ────────────────────────────────────
                elif key == "h":
                    _say("→ home")
                    r.home(duration=2.5, wait=True)
                    target = r.get_pose()
                elif key == "0":
                    _say("→ all-zero joints")
                    r.move_to_joints(np.zeros(6), gripper=0.0,
                                     duration=2.5, wait=True)
                    target = r.get_pose()
                elif key == " ":
                    _say("→ halt")
                    target = r.get_pose()
                    # Send a zero-distance move so any in-flight trajectory is
                    # cancelled and the arm holds where it is.
                    r.move_to(target, duration=0.2, wait=False)

                # ── step adjustment ───────────────────────────────────
                elif key == "+" or key == "=":
                    trans_step = min(0.1, trans_step * 1.5)
                    _say(f"trans step = {trans_step*1000:.0f} mm")
                elif key == "-" or key == "_":
                    trans_step = max(0.001, trans_step / 1.5)
                    _say(f"trans step = {trans_step*1000:.0f} mm")
                elif key == "." or key == ">":
                    rot_step = min(math.radians(45), rot_step * 1.5)
                    _say(f"rot step = {math.degrees(rot_step):.1f} deg")
                elif key == "," or key == "<":
                    rot_step = max(math.radians(0.5), rot_step / 1.5)
                    _say(f"rot step = {math.degrees(rot_step):.1f} deg")

                # ── translation ───────────────────────────────────────
                elif key == "w":
                    target = target.with_gripper(target.gripper)  # noqa (keep type)
                    target = Pose(target.x + trans_step, target.y, target.z,
                                  target.roll, target.pitch, target.yaw,
                                  target.gripper)
                    dirty = True
                elif key == "s":
                    target = Pose(target.x - trans_step, target.y, target.z,
                                  target.roll, target.pitch, target.yaw,
                                  target.gripper)
                    dirty = True
                elif key == "a":
                    target = Pose(target.x, target.y + trans_step, target.z,
                                  target.roll, target.pitch, target.yaw,
                                  target.gripper)
                    dirty = True
                elif key == "d":
                    target = Pose(target.x, target.y - trans_step, target.z,
                                  target.roll, target.pitch, target.yaw,
                                  target.gripper)
                    dirty = True
                elif key == "r":
                    target = Pose(target.x, target.y, target.z + trans_step,
                                  target.roll, target.pitch, target.yaw,
                                  target.gripper)
                    dirty = True
                elif key == "f":
                    target = Pose(target.x, target.y, target.z - trans_step,
                                  target.roll, target.pitch, target.yaw,
                                  target.gripper)
                    dirty = True

                # ── rotation ──────────────────────────────────────────
                elif key == "u":
                    target = Pose(target.x, target.y, target.z,
                                  target.roll + rot_step, target.pitch, target.yaw,
                                  target.gripper)
                    dirty = True
                elif key == "o":
                    target = Pose(target.x, target.y, target.z,
                                  target.roll - rot_step, target.pitch, target.yaw,
                                  target.gripper)
                    dirty = True
                elif key == "i":
                    target = Pose(target.x, target.y, target.z,
                                  target.roll, target.pitch + rot_step, target.yaw,
                                  target.gripper)
                    dirty = True
                elif key == "k":
                    target = Pose(target.x, target.y, target.z,
                                  target.roll, target.pitch - rot_step, target.yaw,
                                  target.gripper)
                    dirty = True
                elif key == "j":
                    target = Pose(target.x, target.y, target.z,
                                  target.roll, target.pitch, target.yaw + rot_step,
                                  target.gripper)
                    dirty = True
                elif key == "l":
                    target = Pose(target.x, target.y, target.z,
                                  target.roll, target.pitch, target.yaw - rot_step,
                                  target.gripper)
                    dirty = True

                # ── gripper (continuous) ──────────────────────────────
                elif key in ("n", "["):
                    target = target.with_gripper(target.gripper - grip_step)
                    dirty = True
                elif key in ("m", "]"):
                    target = target.with_gripper(target.gripper + grip_step)
                    dirty = True
                elif key == ";":
                    grip_step = max(0.001, grip_step / 1.5)
                    _say(f"grip step = {grip_step:.3f} rad")
                elif key == "'":
                    grip_step = min(1.0, grip_step * 1.5)
                    _say(f"grip step = {grip_step:.3f} rad")
                elif key == "G":
                    cur = r.get_pose()
                    _say("")
                    _say(f"current gripper: {cur.gripper:+.3f} rad")
                    try:
                        with raw.cooked():
                            entered = input(
                                "Enter absolute gripper value in radians "
                                "(blank to cancel) > "
                            ).strip()
                    except (EOFError, KeyboardInterrupt):
                        entered = ""
                    if not entered:
                        _say("Gripper set cancelled.")
                    else:
                        try:
                            value = float(entered)
                        except ValueError:
                            _say(f"Invalid number: {entered!r}")
                        else:
                            target = target.with_gripper(value)
                            dirty = True

                else:
                    # Unknown key — ignore quietly.
                    continue

                if dirty:
                    target = _clamp_pose(target)
                    ok = r.move_to(
                        target,
                        duration=MOVE_DURATION,
                        wait=False,
                        timeout=MOVE_TIMEOUT,
                    )
                    status = (
                        f"xyz=[{target.x:+.3f},{target.y:+.3f},{target.z:+.3f}] m "
                        f"rpy=[{target.roll:+.2f},{target.pitch:+.2f},{target.yaw:+.2f}] rad "
                        f"grip={target.gripper:+.3f} "
                        f"{'OK' if ok else 'IK?'}"
                    )
                    if status != last_status:
                        _say(status)
                        last_status = status

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
