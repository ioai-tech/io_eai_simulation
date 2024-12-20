from typing import Dict
import pybullet as p
from .pybullet_env import PyBulletEnvironment


class PhysicsController:
    def __init__(self, env: PyBulletEnvironment):
        self.env = env

    def set_joint_position(self, joint_name: str, position: float):
        """Set joint position"""
        if self.env.robot_id is None:
            return

        for joint_idx in range(p.getNumJoints(self.env.robot_id)):
            joint_info = p.getJointInfo(self.env.robot_id, joint_idx)
            if joint_info[1].decode("utf-8") == joint_name:
                p.setJointMotorControl2(
                    self.env.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=position,
                )
                break

    def set_object_pose(
        self,
        name: str,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ):
        """Set object pose"""
        if name in self.env.objects:
            p.resetBasePositionAndOrientation(
                self.env.objects[name], position, orientation
            )
