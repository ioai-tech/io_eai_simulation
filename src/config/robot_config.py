from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from .camera_config import CameraConfig

# Handle different Python versions
try:
    from typing import Tuple
except ImportError:
    from builtins import tuple as Tuple


@dataclass
class RobotConfig:
    """Robot configuration parameters"""

    urdf_path: Path
    initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    use_fixed_base: bool = True
    camera_config: Optional[CameraConfig] = None
    scale: float = 1.0
