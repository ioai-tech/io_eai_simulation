from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CameraConfig:
    """Camera configuration parameters"""

    joint_name: str = "J_head_yaw"
    fov: float = 1.0
    aspect: float = 1.33
    height: int = 480
    width: int = 640
    offset_transform: Optional[np.ndarray] = None
