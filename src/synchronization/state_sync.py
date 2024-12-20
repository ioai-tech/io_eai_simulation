from typing import Optional
import numpy as np
from ..simulation.pybullet_env import PyBulletEnvironment
from ..visualization.viser_env import ViserEnvironment
from ..visualization.robot_visualization import RobotVisualization
from ..visualization.camera_manager import CameraManager
from ..utils.transforms import convert_quaternion_pybullet_to_viser


class StateSynchronizer:
    def __init__(
        self,
        pybullet_env: PyBulletEnvironment,
        viser_env: ViserEnvironment,
        robot_viz: RobotVisualization,
        camera_manager: CameraManager,
    ):
        self.pybullet_env = pybullet_env
        self.viser_env = viser_env
        self.robot_viz = robot_viz
        self.camera_manager = camera_manager

    def sync_joint_states(self):
        """Synchronize robot joint states from PyBullet to Viser"""
        joint_states = self.pybullet_env.get_joint_states()

        # Update visualization
        self.robot_viz.update_joint_states(joint_states)

        self.camera_manager.update_camera_pose()

    def sync_object_poses(self):
        """Synchronize object poses from PyBullet to Viser"""
        object_poses = self.pybullet_env.get_object_poses()

        for name, (position, orientation) in object_poses.items():
            viser_quaternion = convert_quaternion_pybullet_to_viser(orientation)
            self.viser_env.update_object_pose(name, position, viser_quaternion)

    def sync_robot_root_pose(self):
        """Synchronize robot root pose from PyBullet to Viser"""
        if hasattr(self.pybullet_env, "get_robot_root_pose"):
            position, orientation = self.pybullet_env.get_robot_root_pose()
            viser_quaternion = convert_quaternion_pybullet_to_viser(orientation)
            self.robot_viz.update_root_pose(position, viser_quaternion)

    def sync_all(self):
        """Synchronize all states between PyBullet and Viser"""
        self.sync_joint_states()
        self.sync_object_poses()
        self.sync_robot_root_pose()
