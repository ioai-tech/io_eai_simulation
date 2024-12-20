import numpy as np
from typing import Tuple


def convert_quaternion_pybullet_to_viser(
    pybullet_quat: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """Convert quaternion from PyBullet (x,y,z,w) to Viser format (w,x,y,z)"""
    return (pybullet_quat[3], pybullet_quat[0], pybullet_quat[1], pybullet_quat[2])


# robot_control/utils/path_utils.py
from pathlib import Path
from typing import Dict, Tuple


def get_corresponding_files(
    ply_dir: Path, obj_dir: Path
) -> Dict[str, Tuple[Path, Path]]:
    """Get corresponding PLY and OBJ files"""
    file_pairs = {}
    for ply_file in ply_dir.glob("*.ply"):
        obj_file = obj_dir / f"{ply_file.stem}.obj"
        if obj_file.exists():
            file_pairs[ply_file.stem] = (ply_file, obj_file)
    return file_pairs
