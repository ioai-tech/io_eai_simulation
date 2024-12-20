# viser_env.py

import viser
from typing import Dict, Any
from pathlib import Path
from ..data_loading.gaussian_loader import SplatData
from ..config.robot_config import RobotConfig

try:
    from typing import Tuple
except ImportError:
    from builtins import tuple as Tuple


class ViserEnvironment:
    def __init__(self, config: RobotConfig, port: int = 8080):
        self.config = config
        self.server = viser.ViserServer(port=port)
        self.server.gui.configure_theme(dark_mode=True)
        self.server.scene.world_axes.visible = False

        self.objects: Dict[str, Any] = {}
        self.gui_elements: Dict[str, Any] = {}

    def add_gaussian_splat(
        self,
        name: str,
        splat_data: SplatData,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
    ):
        """Add Gaussian splat visualization"""
        handle = self.server.scene.add_gaussian_splats(
            f"/{name}/gaussian_splats",
            centers=splat_data.centers,
            rgbs=splat_data.rgbs,
            opacities=splat_data.opacities,
            covariances=splat_data.covariances,
            position=position,
            wxyz=orientation,
        )
        self.objects[name] = handle
        return handle

    def update_object_pose(
        self,
        name: str,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
    ):
        """Update object pose"""
        if name in self.objects:
            self.objects[name].position = position
            self.objects[name].wxyz = orientation

    def add_joint_slider(
        self,
        name: str,
        min_val: float,
        max_val: float,
        initial_val: float = 0.0,
        step: float = 0.01,
    ):
        """Add joint control slider"""
        slider = self.server.gui.add_slider(
            label=name, min=min_val, max=max_val, step=step, initial_value=initial_val
        )
        self.gui_elements[name] = slider
        return slider
