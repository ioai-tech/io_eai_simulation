# pybullet_env.py

import pybullet as p
import pybullet_data
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..config.robot_config import RobotConfig

try:
    from typing import Tuple
except ImportError:
    from builtins import tuple as Tuple


class PyBulletEnvironment:
    def __init__(self, config: RobotConfig, gui: bool = True):
        self.config = config
        self.robot_id: Optional[int] = None
        self.objects: Dict[str, int] = {}

        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.load_robot()

    def load_robot(self):
        """Load robot from URDF"""
        self.robot_id = p.loadURDF(
            str(self.config.urdf_path),
            self.config.initial_position,
            (0.0, 0.0, 0.0, 1.0),
            useFixedBase=self.config.use_fixed_base,
        )

    def set_joint_position(self, joint_name: str, position: float):
        """Set position for a specific joint"""
        if self.robot_id is None:
            return

        for joint_idx in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, joint_idx)
            if joint_info[1].decode("utf-8") == joint_name:
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=position,
                    maxVelocity=5.0,  # Adjust as needed
                    force=500.0,  # Adjust as needed
                )
                break

    def get_joint_limits(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Get joint limits for all controllable joints"""
        limits = {}
        if self.robot_id is not None:
            for joint_idx in range(p.getNumJoints(self.robot_id)):
                joint_info = p.getJointInfo(self.robot_id, joint_idx)
                joint_name = joint_info[1].decode("utf-8")
                if joint_info[2] != p.JOINT_FIXED:
                    limits[joint_name] = (joint_info[8], joint_info[9])
        return limits

    def get_joint_states(self) -> Dict[str, float]:
        """Get current joint positions"""
        joint_states = {}
        if self.robot_id is not None:
            for joint_idx in range(p.getNumJoints(self.robot_id)):
                joint_info = p.getJointInfo(self.robot_id, joint_idx)
                joint_name = joint_info[1].decode("utf-8")
                if joint_info[2] != p.JOINT_FIXED:
                    joint_state = p.getJointState(self.robot_id, joint_idx)
                    joint_states[joint_name] = joint_state[0]
        return joint_states

    def add_object(
        self,
        name: str,
        collision_shape: int,
        visual_shape: int,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
        fixed: bool = False,
    ):
        """Add object to environment"""
        mass = 0.0 if fixed else 0.3
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation,
        )
        self.objects[name] = body_id
        return body_id

    def get_object_poses(
        self,
    ) -> Dict[
        str, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]
    ]:
        """Get all object poses"""
        poses = {}
        for name, body_id in self.objects.items():
            pos, orn = p.getBasePositionAndOrientation(body_id)
            poses[name] = (pos, orn)
        return poses

    def step(self):
        """Step physics simulation"""
        p.stepSimulation()

    def reset(self):
        """Reset robot to initial configuration"""
        if self.robot_id is not None:
            for joint_idx in range(p.getNumJoints(self.robot_id)):
                p.resetJointState(self.robot_id, joint_idx, 0.0)

    def __del__(self):
        """Cleanup"""
        if hasattr(self, "physics_client") and p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)
