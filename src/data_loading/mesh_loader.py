from pathlib import Path
from typing import Dict, Tuple
import pybullet as p


class MeshLoader:
    @staticmethod
    def load_obj_file(obj_path: Path) -> Tuple[int, int]:
        """Load OBJ file and create collision and visual shapes"""
        collision_shape = p.createCollisionShape(p.GEOM_MESH, fileName=str(obj_path))
        visual_shape = p.createVisualShape(p.GEOM_MESH, fileName=str(obj_path))
        return collision_shape, visual_shape

    @staticmethod
    def load_objects_from_directory(mesh_dir: Path) -> Dict[str, Tuple[int, int]]:
        """Load all OBJ files from directory"""
        shapes = {}
        for obj_file in mesh_dir.glob("*.obj"):
            shapes[obj_file.stem] = MeshLoader.load_obj_file(obj_file)
        return shapes
