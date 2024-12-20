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
