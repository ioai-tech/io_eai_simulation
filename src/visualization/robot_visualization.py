from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from viser import ViserServer
from viser.extras import ViserUrdf
from ..config.robot_config import RobotConfig


class RobotVisualization:
    def __init__(self, server: ViserServer, config: RobotConfig):
        self.server = server
        self.config = config

        # Create robot root frame
        self.root_frame = self.server.scene.add_frame(
            "/robot_root_frame",
            show_axes=False,
            wxyz=np.array(self.config.initial_orientation),
            position=np.array(self.config.initial_position),
        )

        # Add transform controls
        self.root_transform_controls = self.server.scene.add_transform_controls(
            "/robot_root_frame",
            position=self.root_frame.position,
            wxyz=self.root_frame.wxyz,
        )

        # Initialize URDF
        self.viser_urdf = ViserUrdf(
            server,
            urdf_or_path=self.config.urdf_path,
            scale=self.config.scale,
            root_node_name="/robot_root_frame",
        )

        self._setup_root_frame_callback()

    def update_joint_states(self, joint_states: dict):
        """Update all joint states from PyBullet

        Args:
            joint_states: Dictionary mapping joint names to their current positions
        """
        # Update URDF configuration
        self.viser_urdf.update_cfg(joint_states)

    def update_root_pose(self, position: np.ndarray, orientation: np.ndarray):
        """Update root frame pose"""
        self.root_frame.position = position
        self.root_frame.wxyz = orientation

    def _setup_root_frame_callback(self):
        """Set up root frame transform callback"""

        def update_root_frame(transform):
            # Update root frame position and orientation
            self.root_frame.position = transform.position
            self.root_frame.wxyz = transform.wxyz

            # # Update camera pose if camera manager exists
            # if hasattr(self, "camera_manager"):
            #     self.camera_manager.update_camera_pose()

        # Listen for transform control changes
        self.root_transform_controls.on_update(update_root_frame)

    def reset_state(self):
        """Reset robot to initial state"""
        self.root_frame.position = np.array(self.config.initial_position)
        self.root_frame.wxyz = np.array(self.config.initial_orientation)
