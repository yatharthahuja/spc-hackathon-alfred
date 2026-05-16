"""End-effector pose + trajectory control for the reBot Arm B601-DM.

Public surface
--------------
- ``Pose``               — dataclass: x, y, z, roll, pitch, yaw, gripper.
- ``Tolerances``         — dataclass: position / orientation / gripper tolerances.
- ``Robot``              — connects to the arm (6 joints) and the gripper (motor 0x07)
                           over a single shared damiao serial bus, runs a 500 Hz
                           POS_VEL control loop in the background, and exposes:
      * ``home(...)``                    — go to the captured home pose
                                            (start pose + ``home_offset``, default
                                            ``(0, 0, +0.2 m)`` so it lifts straight
                                            up from wherever the EE was at connect).
      * ``home_pose`` / ``set_home_pose`` — read or override the captured home pose.
      * ``move_to(pose, ...)``           — IK + smooth Cartesian (SE(3)) trajectory.
      * ``move_to_joints(q, ...)``       — direct joint-space move (linear interp,
                                            no IK), useful for known safe configs
                                            such as the all-zero rest pose.
      * ``get_pose()``                   — current end-effector Pose (from FK + gripper).
      * ``is_at(pose, ...)``             — check if the EE has reached a pose.
      * ``execute_trajectory(poses, ...)`` — visit each Pose; only sends the next
                                             waypoint after the previous one is
                                             reached (or aborts on timeout). When
                                             done (success OR abort), the arm
                                             returns to all-zero joints by default
                                             — pass ``return_to_zero=False`` to skip.
      * ``disable()``                    — stop the control loop and disable all
                                            motors *in place* (no motion).
                                            ``disconnect()`` / leaving a ``with``
                                            block also disables and closes the bus.

Hardware assumptions
--------------------
Matches the bundled reBotArm_control_py defaults:
  - Damiao motors on /dev/ttyACM0 @ 921600.
  - Arm joints 0x01..0x06 (DM4340P x3 + DM4310 x3), feedback ids 0x11..0x16.
  - Gripper motor 0x07, feedback id 0x17, model 4310.
  - URDF: reBotArm_control_py/urdf/.../reBot-DevArm_fixend.urdf (loaded via Pinocchio).

Make sure /dev/ttyACM0 is readable/writable (`sudo chmod 666 /dev/ttyACM0` or add
yourself to the `dialout` group) and that the motorbridge gateway is NOT running
(it would hold the serial port).

Quick start
-----------
    from trajectory import Robot, Pose

    with Robot() as r:
        r.home()
        print("at:", r.get_pose())
        r.move_to(Pose(x=0.25, y=0.05, z=0.18, gripper=0.0))
        r.execute_trajectory([
            Pose(x=0.25, y=0.05, z=0.18, gripper=0.0),
            Pose(x=0.25, y=0.05, z=0.18, gripper=0.6),
            Pose(x=0.22, y=-0.05, z=0.20, gripper=0.6),
            Pose(x=0.22, y=-0.05, z=0.20, gripper=0.0),
        ])

Run directly for a self-contained demo:
    python trajectory.py
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

REPO_DIR = Path(__file__).resolve().parent / "reBotArm_control_py"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from motorbridge import Mode  # noqa: E402

from reBotArm_control_py.actuator import RobotArm  # noqa: E402
from reBotArm_control_py.kinematics import (  # noqa: E402
    compute_fk,
    get_end_effector_frame_id,
    joint_to_pose,
    load_robot_model,
    pos_rot_to_se3,
)
from reBotArm_control_py.kinematics.inverse_kinematics import (  # noqa: E402
    IKParams,
    solve_ik,
    solve_ik_with_retry,
)
from reBotArm_control_py.trajectory import (  # noqa: E402
    IKParams as ClikIKParams,
    TrajPlanParams,
    TrajProfile,
    plan_cartesian_geodesic_trajectory,
    track_trajectory,
)


HOME_JOINTS: np.ndarray = np.zeros(6, dtype=np.float64)
HOME_GRIPPER: float = 0.0
DEFAULT_HOME_OFFSET: tuple[float, float, float] = (0.0, 0.0, 0.2)


# ──────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Pose:
    """End-effector pose in the robot base frame.

    Position is metres. Orientation is XYZ Euler angles (roll about X,
    pitch about Y, yaw about Z) in radians. ``gripper`` is the gripper
    joint angle in motor radians (calibrate yourself: typically 0.0 at
    the "set zero" reference; positive values close the jaws).
    """

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    gripper: float = 0.0

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def rpy(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.yaw], dtype=np.float64)

    def with_gripper(self, gripper: float) -> "Pose":
        return replace(self, gripper=gripper)

    def __str__(self) -> str:
        return (
            f"Pose(xyz=[{self.x:+.3f},{self.y:+.3f},{self.z:+.3f}] m, "
            f"rpy=[{self.roll:+.2f},{self.pitch:+.2f},{self.yaw:+.2f}] rad, "
            f"grip={self.gripper:+.3f})"
        )


@dataclass
class Tolerances:
    """Acceptance bounds used by ``is_at`` / ``execute_trajectory``."""

    pos: float = 0.005          # meters
    orient_deg: float = 2.0     # degrees of geodesic rotation distance
    gripper: float = 0.05       # radians
    joint: float = 0.02         # radians (used internally for joint settle)


# ──────────────────────────────────────────────────────────────────────
# Robot
# ──────────────────────────────────────────────────────────────────────

class Robot:
    """High-level controller for the reBot Arm B601-DM.

    Internally:
      - Wraps ``reBotArm_control_py.actuator.RobotArm`` for the 6 arm joints.
      - Attaches the gripper motor (0x07) to the same damiao motorbridge
        ``Controller``, so a single serial port is opened.
      - Runs the arm's high-rate POS_VEL control loop and pumps the
        gripper command from the same callback.

    Construct, ``connect()`` (or use as a context manager), and call
    ``home`` / ``move_to`` / ``execute_trajectory`` / ``get_pose``.
    """

    def __init__(
        self,
        arm_cfg: Optional[str] = None,
        gripper_motor_id: int = 0x07,
        gripper_feedback_id: int = 0x17,
        gripper_model: str = "4310",
        gripper_vlim: float = 3.0,
        gripper_kp_pos: float = 50.0,
        gripper_ki_pos: float = 1.0,
        gripper_kp_vel: float = 0.0008,
        gripper_ki_vel: float = 0.002,
        ik_max_iter: int = 200,
        ik_tol: float = 1e-4,
        plan_dt: float = 0.02,
        home_offset: tuple[float, float, float] = DEFAULT_HOME_OFFSET,
    ) -> None:
        if arm_cfg is None:
            arm_cfg = str(REPO_DIR / "config" / "arm.yaml")

        self.arm = RobotArm(arm_cfg)

        damiao_ctrl = self.arm._ctrl_map.get("damiao")
        if damiao_ctrl is None:
            raise RuntimeError(
                "Arm did not initialise a damiao controller; cannot attach gripper"
            )
        self._gripper_mot = damiao_ctrl.add_damiao_motor(
            gripper_motor_id, gripper_feedback_id, gripper_model,
        )
        self._gripper_vlim = float(gripper_vlim)
        self._gripper_pi = (
            float(gripper_kp_vel), float(gripper_ki_vel),
            float(gripper_kp_pos), float(gripper_ki_pos),
        )

        self._model = load_robot_model()
        self._data = self._model.createData()
        self._ee_frame_id = get_end_effector_frame_id(self._model)
        self._n_joints = self._model.nq
        if self._n_joints != self.arm.num_joints:
            raise RuntimeError(
                f"URDF DOF ({self._n_joints}) does not match arm.yaml joints "
                f"({self.arm.num_joints})."
            )

        self._joint_target = np.zeros(self._n_joints, dtype=np.float64)
        self._gripper_target = 0.0
        self._joint_vlim = np.array(
            [j.vlim for j in self.arm._joints], dtype=np.float64,
        )
        self._joint_vlim_override: Optional[np.ndarray] = None

        self._ik_params = IKParams(
            max_iter=ik_max_iter, tolerance=ik_tol,
            step_size=0.5, damping=1e-6,
        )
        self._clik_params = ClikIKParams(
            max_iter=ik_max_iter, tolerance=ik_tol,
            damping=1e-6, step_size=0.8,
        )

        self._plan_dt = float(plan_dt)
        self._traj_profile = TrajProfile.MIN_JERK

        # Home pose is captured on connect() as (start_pose + home_offset).
        # Until then it's None; calling home() before connect() raises.
        self._home_offset = tuple(float(v) for v in home_offset)
        self._start_pose: Optional[Pose] = None
        self._home_pose: Optional[Pose] = None

        self._traj_lock = threading.Lock()
        self._traj_q_pts: List[np.ndarray] = []
        self._traj_grip_pts: List[float] = []
        self._traj_duration_s: float = 0.0
        # gripper_after: if True the gripper is held at the start value during
        # the joint motion and only commanded to ``_traj_final_grip`` *after*
        # the joints have physically arrived within ``_traj_joint_settle_tol``
        # (polled inside the send thread). ``_traj_settle_timeout`` is a
        # safety cap so we still fire the gripper even if the joints never
        # settle.
        self._traj_gripper_after: bool = False
        self._traj_final_grip: float = 0.0
        self._traj_joint_settle_tol: float = 0.02
        self._traj_settle_timeout: float = 3.0
        self._send_thread: Optional[threading.Thread] = None
        self._stop_send = threading.Event()
        self._moving = False

        self._connected = False

    # ── connection ───────────────────────────────────────────────────

    def connect(self) -> None:
        if self._connected:
            return
        self._configure_gripper_pos_vel()
        self.arm.mode_pos_vel()
        self.arm.enable()

        q, _, _ = self.arm.get_state()
        self._joint_target[:] = q
        try:
            st = self._gripper_mot.get_state()
            if st is not None:
                self._gripper_target = float(st.pos)
        except Exception:
            pass

        self.arm.start_control_loop(self._control_cb)
        time.sleep(0.1)
        self._connected = True

        # Capture the start pose (FK from current joints + gripper) and define
        # home as (start_pose + home_offset), keeping orientation and gripper.
        self._start_pose = self.get_pose()
        dx, dy, dz = self._home_offset
        self._home_pose = Pose(
            x=self._start_pose.x + dx,
            y=self._start_pose.y + dy,
            z=self._start_pose.z + dz,
            roll=self._start_pose.roll,
            pitch=self._start_pose.pitch,
            yaw=self._start_pose.yaw,
            gripper=self._start_pose.gripper,
        )

    def disable(self) -> None:
        """Stop the control loop and disable all motors (keeps the bus open).

        Useful for parking the arm in place at the end of an operation
        without tearing down the connection. After this, you must call
        ``connect()`` again before sending more motion commands.
        """
        if not self._connected:
            return
        self._stop_send.set()
        if self._send_thread is not None:
            self._send_thread.join(timeout=1.0)
            self._send_thread = None
        # arm.disable() internally stops the control loop and then calls
        # disable_all() on every controller, which also disables the gripper
        # (it lives on the same damiao controller).
        self.arm.disable()

    def disconnect(self) -> None:
        """Disable all motors and close the serial bus.

        Does **not** move the arm anywhere — whatever joint configuration
        the arm is in when this is called is where it ends up. If you want
        to land at a specific pose first, call ``home()`` or
        ``move_to_joints(...)`` explicitly before exiting the ``with`` block.
        """
        if not self._connected:
            return
        self._stop_send.set()
        if self._send_thread is not None:
            self._send_thread.join(timeout=1.0)
            self._send_thread = None
        # arm.disconnect() = stop_control_loop -> disable_all -> shutdown -> close
        self.arm.disconnect()
        self._connected = False

    def __enter__(self) -> "Robot":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    # ── low-level setup helpers ──────────────────────────────────────

    def _configure_gripper_pos_vel(self) -> None:
        vkp, vki, pkp, pki = self._gripper_pi
        try:
            self._gripper_mot.write_register_f32(25, vkp)
            self._gripper_mot.write_register_f32(26, vki)
            self._gripper_mot.write_register_f32(27, pkp)
            self._gripper_mot.write_register_f32(28, pki)
            time.sleep(0.02)
            self._gripper_mot.ensure_mode(Mode.POS_VEL, 1000)
        except Exception as e:
            print(f"[Robot] gripper POS_VEL config failed: {e}")

    def _control_cb(self, arm: RobotArm, dt: float) -> None:
        vlim = (
            self._joint_vlim_override
            if self._joint_vlim_override is not None
            else self._joint_vlim
        )
        arm.pos_vel(self._joint_target, vlim=vlim)
        try:
            self._gripper_mot.send_pos_vel(
                float(self._gripper_target), float(self._gripper_vlim),
            )
        except Exception:
            pass
        try:
            self._gripper_mot.request_feedback()
        except Exception:
            pass

    # ── state read-back ──────────────────────────────────────────────

    def get_joint_positions(self) -> np.ndarray:
        q, _, _ = self.arm.get_state()
        return q

    def get_gripper_position(self) -> float:
        try:
            st = self._gripper_mot.get_state()
            if st is not None:
                return float(st.pos)
        except Exception:
            pass
        return float(self._gripper_target)

    def get_pose(self) -> Pose:
        q = self.get_joint_positions()
        pos, rpy = joint_to_pose(q)
        grip = self.get_gripper_position()
        return Pose(
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
            roll=float(rpy[0]), pitch=float(rpy[1]), yaw=float(rpy[2]),
            gripper=grip,
        )

    # ── inverse kinematics ───────────────────────────────────────────

    def _ik(self, pose: Pose, q_seed: np.ndarray) -> tuple[bool, np.ndarray]:
        target = pos_rot_to_se3(
            pose.position(),
            roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw,
        )
        result = solve_ik(
            self._model, self._data, self._ee_frame_id,
            target, q_seed.copy(), self._ik_params,
        )
        if not result.success:
            result = solve_ik_with_retry(
                self._model, self._data, self._ee_frame_id,
                target, q_seed.copy(), self._ik_params,
                max_retries=8,
            )
        return result.success, result.q

    # ── motion primitives ────────────────────────────────────────────

    @property
    def home_pose(self) -> Pose:
        """The Cartesian home pose (captured at connect()).

        Defined as ``start_pose + home_offset`` (default offset is +0.2 m in Z,
        i.e. straight up from wherever the EE was when ``connect()`` ran).
        """
        if self._home_pose is None:
            raise RuntimeError("home_pose is not defined until connect() runs")
        return self._home_pose

    def set_home_pose(self, pose: Pose) -> None:
        """Override the home pose (e.g. after a manual jog or recalibration)."""
        self._home_pose = pose

    def home(
        self, *,
        duration: float = 3.0,
        wait: bool = True,
        timeout: float = 15.0,
        gripper: Optional[float] = None,
        tolerances: Optional[Tolerances] = None,
    ) -> bool:
        """Move the EE to the home Cartesian pose captured at connect().

        Pass ``gripper`` to override the home gripper value (otherwise it
        uses whatever was active at connect time, or whatever
        ``set_home_pose`` was given).
        """
        target = self.home_pose
        if gripper is not None:
            target = target.with_gripper(float(gripper))
        return self.move_to(
            target,
            duration=duration,
            wait=wait,
            timeout=timeout,
            tolerances=tolerances,
        )

    def move_to(
        self,
        pose: Pose,
        *,
        duration: float = 2.0,
        wait: bool = True,
        timeout: Optional[float] = None,
        tolerances: Optional[Tolerances] = None,
        gripper_after: bool = False,
    ) -> bool:
        """Plan + execute a smooth Cartesian motion to ``pose``.

        Returns ``True`` only if the EE actually settles within
        ``tolerances`` (when ``wait=True``).

        Parameters
        ----------
        gripper_after : bool, default False
            If True, the gripper is held at its current value through the
            joint motion and only commanded to ``pose.gripper`` after the
            joint setpoints finish (use for pick/place-style sequencing).
            If False (default), the gripper is linearly interpolated
            alongside the joints — preferred for teleop responsiveness.
        """
        if not self._connected:
            raise RuntimeError("Robot not connected. Call connect() or use 'with'.")

        q_seed = self.get_joint_positions()
        ok, q_goal = self._ik(pose, q_seed)
        if not ok:
            print(f"[Robot] IK failed for {pose}")
            return False

        if timeout is None:
            timeout = max(duration * 2.5, 5.0)

        tol = tolerances or Tolerances()
        moved = self._goto_joints(
            q_goal, pose.gripper,
            duration=duration, wait=wait, timeout=timeout,
            joint_tol=tol.joint, gripper_tol=tol.gripper,
            gripper_after=gripper_after,
        )
        if not wait:
            return moved
        # Even after joints settle, verify the actual Cartesian pose
        return moved and self.is_at(pose, tolerances=tol)

    def move_to_joints(
        self,
        q: "Sequence[float] | np.ndarray",
        *,
        gripper: Optional[float] = None,
        duration: float = 2.0,
        wait: bool = True,
        timeout: Optional[float] = None,
        tolerances: Optional[Tolerances] = None,
        gripper_after: bool = True,
    ) -> bool:
        """Move directly to a target joint configuration (linear joint interp).

        Unlike ``move_to(Pose)`` this skips IK and Cartesian planning, so it's
        the right tool for known-safe joint targets (e.g. the all-zero rest
        pose) and for situations where IK might fail (near singularities,
        outside the workspace).

        Parameters
        ----------
        q : array-like of shape (n_joints,)
            Target joint angles in radians.
        gripper : float, optional
            Target gripper angle (rad). ``None`` (default) holds the current
            gripper position.
        duration, wait, timeout, tolerances : see ``move_to``.
        gripper_after : bool, default True
            If True (default), the gripper holds at its current value during
            the joint motion and only moves to ``gripper`` after the joints
            have reached the target. Set False to interpolate the gripper
            linearly through the motion instead.
        """
        if not self._connected:
            raise RuntimeError("Robot not connected. Call connect() or use 'with'.")
        q_goal = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_goal.shape != (self._n_joints,):
            raise ValueError(
                f"q must have shape ({self._n_joints},), got {q_goal.shape}"
            )
        grip_goal = (
            float(gripper) if gripper is not None
            else self.get_gripper_position()
        )
        if timeout is None:
            timeout = max(duration * 2.5, 5.0)
        tol = tolerances or Tolerances()
        return self._goto_joints(
            q_goal, grip_goal,
            duration=duration, wait=wait, timeout=timeout,
            joint_tol=tol.joint, gripper_tol=tol.gripper,
            interpolate="joint",
            gripper_after=gripper_after,
        )

    def execute_trajectory(
        self,
        poses: Sequence[Pose],
        *,
        segment_duration: float = 2.0,
        timeout_per_segment: Optional[float] = None,
        tolerances: Optional[Tolerances] = None,
        verbose: bool = True,
        return_to_zero: bool = True,
        zero_duration: float = 3.0,
        zero_gripper: float = 0.0,
    ) -> bool:
        """Visit each pose in order; wait for each to be reached.

        The next waypoint is only sent after the current one is reached
        within ``tolerances`` (default: 5 mm position, 2 deg orientation,
        0.05 rad gripper). Returns ``True`` iff every waypoint succeeds.

        After the trajectory finishes (whether all waypoints succeeded or
        the loop aborted on a failed waypoint), the arm is returned to the
        all-zero joint configuration (and gripper = ``zero_gripper``). Set
        ``return_to_zero=False`` to skip this.
        """
        if not self._connected:
            raise RuntimeError("Robot not connected. Call connect() or use 'with'.")
        if timeout_per_segment is None:
            timeout_per_segment = max(segment_duration * 2.5, 5.0)
        tol = tolerances or Tolerances()
        n = len(poses)
        success = True
        try:
            for i, p in enumerate(poses):
                if verbose:
                    print(f"[traj] {i+1}/{n} -> {p}")
                ok = self.move_to(
                    p,
                    duration=segment_duration,
                    wait=True,
                    timeout=timeout_per_segment,
                    tolerances=tol,
                )
                if not ok:
                    if verbose:
                        cur = self.get_pose()
                        print(f"[traj] waypoint {i+1} not reached; aborting at {cur}")
                    success = False
                    break
                if verbose:
                    print(f"[traj] {i+1}/{n} reached.")
            return success
        finally:
            if return_to_zero:
                if verbose:
                    print("[traj] returning to all-zero joints "
                          f"(gripper={zero_gripper:+.3f})")
                try:
                    self.move_to_joints(
                        np.zeros(self._n_joints, dtype=np.float64),
                        gripper=zero_gripper,
                        duration=zero_duration,
                        wait=True,
                        tolerances=Tolerances(
                            pos=0.02, orient_deg=10.0,
                            gripper=max(tol.gripper, 0.1),
                            joint=max(tol.joint, 0.05),
                        ),
                    )
                except Exception as e:
                    print(f"[traj] return-to-zero failed: {e}")

    # ── reach checks ─────────────────────────────────────────────────

    def is_at(
        self,
        pose: Pose,
        *,
        tolerances: Optional[Tolerances] = None,
    ) -> bool:
        tol = tolerances or Tolerances()
        cur = self.get_pose()
        dpos = math.sqrt(
            (cur.x - pose.x) ** 2
            + (cur.y - pose.y) ** 2
            + (cur.z - pose.z) ** 2
        )
        R_cur = pos_rot_to_se3(
            np.zeros(3), roll=cur.roll, pitch=cur.pitch, yaw=cur.yaw,
        ).rotation
        R_des = pos_rot_to_se3(
            np.zeros(3), roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw,
        ).rotation
        cos_theta = max(-1.0, min(1.0, 0.5 * (np.trace(R_cur.T @ R_des) - 1.0)))
        ang_deg = math.degrees(math.acos(cos_theta))
        dgrip = abs(cur.gripper - pose.gripper)
        return dpos < tol.pos and ang_deg < tol.orient_deg and dgrip < tol.gripper

    # ── internal: joint-space goto + trajectory streaming ───────────

    def _goto_joints(
        self,
        q_goal: np.ndarray,
        gripper_goal: float,
        *,
        duration: float,
        wait: bool,
        timeout: float,
        joint_tol: float = 0.02,
        gripper_tol: float = 0.05,
        use_vlim_override: Optional[float] = None,
        interpolate: str = "cartesian",
        gripper_after: bool = False,
    ) -> bool:
        q_start = self.get_joint_positions()
        grip_start = self.get_gripper_position()
        duration = max(float(duration), 0.5)

        joint_pts: List[np.ndarray] = []
        if interpolate == "cartesian":
            # Plan an SE(3) geodesic between current and goal pose, then CLIK-track
            T_start = compute_fk(self._model, q_start)[2]
            T_end = compute_fk(self._model, q_goal)[2]
            plan_params = TrajPlanParams(dt=self._plan_dt, profile=self._traj_profile)
            try:
                cart_traj = plan_cartesian_geodesic_trajectory(
                    T_start, T_end, duration, plan_params,
                )
                joint_traj = track_trajectory(
                    self._model, self._ee_frame_id,
                    cart_traj.trajectory, q_start, self._clik_params,
                    null_gain=0.1,
                )
                joint_pts = [pt.q.copy() for pt in joint_traj] if joint_traj else []
            except Exception as e:
                print(f"[Robot] Cartesian plan failed ({e}); falling back to joint interp")
                joint_pts = []
        elif interpolate != "joint":
            raise ValueError(
                f"interpolate must be 'cartesian' or 'joint', got {interpolate!r}"
            )

        if not joint_pts:
            steps = max(int(duration / self._plan_dt), 2)
            joint_pts = [
                q_start + (q_goal - q_start) * (t / (steps - 1))
                for t in range(steps)
            ]

        n = len(joint_pts)
        if gripper_after:
            # Hold gripper at start through the joint motion; the send loop
            # will command the final value once the joint setpoints finish.
            grip_pts = [float(grip_start)] * n
        else:
            grip_pts = [
                grip_start + (gripper_goal - grip_start) * (t / max(n - 1, 1))
                for t in range(n)
            ]

        # Cancel any in-flight stream and start a new one
        self._stop_send.set()
        if self._send_thread is not None:
            self._send_thread.join(timeout=1.0)
        with self._traj_lock:
            self._traj_q_pts = joint_pts
            self._traj_grip_pts = grip_pts
            self._traj_duration_s = duration
            self._traj_gripper_after = bool(gripper_after)
            self._traj_final_grip = float(gripper_goal)
            self._traj_joint_settle_tol = float(joint_tol)
            # Safety cap on how long the send thread will wait for the joints
            # to physically arrive before firing the gripper anyway.
            self._traj_settle_timeout = max(2.0, duration)
        self._stop_send.clear()
        self._moving = True
        self._joint_vlim_override = (
            np.full(self._n_joints, use_vlim_override, dtype=np.float64)
            if use_vlim_override is not None else None
        )
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()

        if not wait:
            return True
        ok = self._wait_joint(
            q_goal, gripper_goal,
            timeout=timeout,
            joint_tol=joint_tol, gripper_tol=gripper_tol,
        )
        self._joint_vlim_override = None
        return ok

    def _send_loop(self) -> None:
        with self._traj_lock:
            pts = list(self._traj_q_pts)
            gpts = list(self._traj_grip_pts)
            duration = self._traj_duration_s
            gripper_after = bool(self._traj_gripper_after)
            final_grip = float(self._traj_final_grip)
            joint_settle_tol = float(self._traj_joint_settle_tol)
            settle_timeout = float(self._traj_settle_timeout)
        n = len(pts)
        if n == 0:
            self._moving = False
            return
        interval = duration / n
        for i in range(n):
            if self._stop_send.is_set():
                self._moving = False
                return
            self._joint_target[:] = pts[i]
            self._gripper_target = float(gpts[i])
            time.sleep(interval)
        # Hold the final joint target briefly to let the PI loop catch up.
        time.sleep(0.05)
        # If the gripper was held during the motion, do NOT update it on a
        # timer — poll the *measured* joint positions and only fire the
        # gripper once the arm has physically arrived at the target (within
        # joint_settle_tol). A safety timeout keeps us from getting stuck.
        if gripper_after and not self._stop_send.is_set():
            q_goal = self._joint_target.copy()
            deadline = time.monotonic() + settle_timeout
            arrived = False
            while time.monotonic() < deadline:
                if self._stop_send.is_set():
                    self._moving = False
                    return
                q_now = self.get_joint_positions()
                if np.max(np.abs(q_now - q_goal)) < joint_settle_tol:
                    arrived = True
                    break
                time.sleep(0.02)
            if not arrived:
                print(
                    "[Robot] gripper_after: joints did not settle within "
                    f"{settle_timeout:.1f}s (tol {joint_settle_tol:.3f} rad); "
                    "firing gripper anyway."
                )
            # Joints are at (or as close as they're going to get to) q_goal.
            # Now command the gripper to its target.
            self._gripper_target = final_grip
        self._moving = False

    def _wait_joint(
        self,
        q_goal: np.ndarray,
        gripper_goal: float,
        *,
        timeout: float,
        joint_tol: float,
        gripper_tol: float,
    ) -> bool:
        deadline = time.monotonic() + timeout
        # First, let the streamer push all setpoints
        while self._moving and time.monotonic() < deadline:
            time.sleep(0.02)
        # Then wait for joints (and gripper) to actually settle
        while time.monotonic() < deadline:
            q = self.get_joint_positions()
            grip = self.get_gripper_position()
            if (
                np.max(np.abs(q - q_goal)) < joint_tol
                and abs(grip - gripper_goal) < gripper_tol
            ):
                return True
            time.sleep(0.02)
        return False


# ──────────────────────────────────────────────────────────────────────
# CLI demo
# ──────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """Tiny demo: lift to home, read pose, then walk a small square around home."""
    print("=== reBot Arm B601-DM trajectory demo ===")
    with Robot() as r:
        print("Start pose:", r.get_pose())
        print("Home pose: ", r.home_pose, "  (start + offset)")
        print("Lifting to home...")
        r.home(duration=3.0)
        print("At home:   ", r.get_pose())

        h = r.home_pose
        # Small square relative to the home Cartesian point.
        waypoints = [
            Pose(x=h.x + 0.00, y=h.y + 0.04, z=h.z,          gripper=0.0),
            Pose(x=h.x + 0.00, y=h.y + 0.04, z=h.z,          gripper=0.5),
            Pose(x=h.x - 0.03, y=h.y - 0.04, z=h.z + 0.02,   gripper=0.5),
            Pose(x=h.x - 0.03, y=h.y - 0.04, z=h.z + 0.02,   gripper=0.0),
            Pose(x=h.x,        y=h.y,        z=h.z,          gripper=0.0),
        ]
        ok = r.execute_trajectory(
            waypoints,
            segment_duration=2.0,
            tolerances=Tolerances(pos=0.01, orient_deg=4.0, gripper=0.1),
        )
        # execute_trajectory auto-returns to all-zero joints on completion.
        print("Trajectory completed." if ok else "Trajectory FAILED.")
        print("Final pose (should be near joint-zero pose):", r.get_pose())


if __name__ == "__main__":
    _demo()
