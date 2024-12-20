# alignment_utils.py

import numpy as np
import pybullet as p
from pathlib import Path
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation as R


class AlignmentUtils:
    @staticmethod
    def load_and_analyze_obj(obj_path: Path):
        """
        加载并分析OBJ文件的尺寸和中心点
        """
        mesh = trimesh.load(str(obj_path))

        # 获取边界框信息
        bounds = mesh.bounds
        dimensions = bounds[1] - bounds[0]
        center = mesh.centroid

        return {"dimensions": dimensions, "center": center, "bounds": bounds}

    @staticmethod
    def load_and_analyze_ply(ply_path: Path):
        """
        加载并分析PLY点云文件的尺寸和中心点
        """
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points)

        # 计算点云的边界框和中心
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        dimensions = max_bound - min_bound
        center = (max_bound + min_bound) / 2

        return {
            "dimensions": dimensions,
            "center": center,
            "bounds": np.vstack([min_bound, max_bound]),
        }

    @staticmethod
    def check_alignment(obj_info, ply_info, threshold=0.1):
        """
        检查OBJ和PLY的对齐情况
        """
        # 检查尺寸差异
        dim_diff = np.abs(obj_info["dimensions"] - ply_info["dimensions"])
        center_diff = np.abs(obj_info["center"] - ply_info["center"])

        is_aligned = {
            "dimensions": all(dim_diff < threshold),
            "center": all(center_diff < threshold),
        }

        differences = {"dimensions": dim_diff, "center": center_diff}

        return is_aligned, differences

    @staticmethod
    def get_object_pose_pybullet(body_id):
        """
        获取PyBullet中物体的位姿
        """
        pos, orn = p.getBasePositionAndOrientation(body_id)
        return np.array(pos), np.array(orn)

    @staticmethod
    def get_object_pose_viser(viser_object):
        """
        获取Viser中物体的位姿
        """
        pos = np.array(viser_object.position)
        orn = np.array(viser_object.wxyz)  # viser使用(w,x,y,z)顺序
        return pos, orn

    @staticmethod
    def compute_transform_matrix(position, orientation):
        """
        计算4x4变换矩阵
        """
        # 将四元数转换为旋转矩阵
        if len(orientation) == 4:
            rot_matrix = R.from_quat(orientation).as_matrix()
        else:
            rot_matrix = orientation

        # 创建4x4变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = position

        return transform

    def verify_object_alignment(self, pybullet_env, viser_env, object_name):
        """
        验证PyBullet和Viser中物体的对齐情况
        """
        # 获取PyBullet中的位姿
        body_id = pybullet_env.objects[object_name]
        pb_pos, pb_orn = self.get_object_pose_pybullet(body_id)
        pb_transform = self.compute_transform_matrix(pb_pos, pb_orn)

        # 获取Viser中的位姿
        viser_obj = viser_env.objects[object_name]
        viser_pos, viser_orn = self.get_object_pose_viser(viser_obj)
        # 转换四元数顺序从(w,x,y,z)到(x,y,z,w)
        viser_orn = np.array([viser_orn[1], viser_orn[2], viser_orn[3], viser_orn[0]])
        viser_transform = self.compute_transform_matrix(viser_pos, viser_orn)

        # 计算位置和旋转的差异
        pos_diff = np.linalg.norm(pb_pos - viser_pos)
        rot_diff = np.abs(
            np.arccos(
                np.clip(
                    np.trace(pb_transform[:3, :3].T @ viser_transform[:3, :3]) / 2 - 1,
                    -1,
                    1,
                )
            )
        )

        return {
            "position_difference": pos_diff,
            "rotation_difference_rad": rot_diff,
            "rotation_difference_deg": np.degrees(rot_diff),
            "is_aligned": pos_diff < 0.01 and rot_diff < np.radians(5),  # 阈值可调整
        }

    @staticmethod
    def suggest_corrections(alignment_result):
        """
        根据对齐检查结果提供修正建议
        """
        suggestions = []

        if alignment_result["position_difference"] >= 0.01:
            suggestions.append(
                f"Position misalignment detected: {alignment_result['position_difference']:.3f} units"
            )

        if alignment_result["rotation_difference_deg"] >= 5:
            suggestions.append(
                f"Rotation misalignment detected: {alignment_result['rotation_difference_deg']:.1f} degrees"
            )

        if not alignment_result["is_aligned"]:
            suggestions.append("Consider adjusting:")
            suggestions.append("1. Check if the coordinate systems match")
            suggestions.append("2. Verify quaternion conventions (w,x,y,z vs x,y,z,w)")
            suggestions.append("3. Check if the models are centered consistently")

        return suggestions
