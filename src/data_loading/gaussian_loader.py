import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import numpy.typing as npt
from plyfile import PlyData
from viser import transforms as tf


@dataclass
class SplatData:
    centers: npt.NDArray[np.floating]
    rgbs: npt.NDArray[np.floating]
    opacities: npt.NDArray[np.floating]
    covariances: npt.NDArray[np.floating]


class GaussianSplatLoader:
    """高斯点数据加载器，支持 .splat 和 .ply 文件格式"""

    @staticmethod
    def load_splat_file(splat_path: Path, center: bool = False) -> SplatData:
        """加载 .splat 格式的高斯点文件"""
        start_time = time.perf_counter()

        splat_buffer = splat_path.read_bytes()
        bytes_per_gaussian = 3 * 4 + 3 * 4 + 4 + 4
        assert len(splat_buffer) % bytes_per_gaussian == 0

        num_gaussians = len(splat_buffer) // bytes_per_gaussian
        splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
            (num_gaussians, bytes_per_gaussian)
        )

        scales = splat_uint8[:, 12:24].copy().view(np.float32)
        wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
        Rs = tf.SO3(wxyzs).as_matrix()

        covariances = np.einsum(
            "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
        )
        centers = splat_uint8[:, 0:12].copy().view(np.float32)

        if center:
            centers -= np.mean(centers, axis=0, keepdims=True)

        print(
            f"Splat file with {num_gaussians=} loaded in {time.perf_counter() - start_time:.4f} seconds"
        )

        return SplatData(
            centers=centers,
            rgbs=splat_uint8[:, 24:27] / 255.0,
            opacities=splat_uint8[:, 27:28] / 255.0,
            covariances=covariances,
        )

    @staticmethod
    def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatData:
        """加载 .ply 格式的高斯点文件"""
        start_time = time.perf_counter()
        SH_C0 = 0.28209479177387814

        plydata = PlyData.read(ply_file_path)
        v = plydata["vertex"]

        positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
        scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
        wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)

        colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
        opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

        Rs = tf.SO3(wxyzs).as_matrix()
        covariances = np.einsum(
            "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
        )

        if center:
            positions -= np.mean(positions, axis=0, keepdims=True)

        num_gaussians = len(v)
        print(
            f"PLY file with {num_gaussians=} loaded in {time.perf_counter() - start_time:.4f} seconds"
        )

        return SplatData(
            centers=positions,
            rgbs=colors,
            opacities=opacities,
            covariances=covariances,
        )
