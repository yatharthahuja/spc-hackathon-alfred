from __future__ import annotations


class RobotArmInterface:
    def move_to_pose(self, pose_name: str) -> None:
        raise NotImplementedError("Real arm control is intentionally not implemented in the MVP.")
