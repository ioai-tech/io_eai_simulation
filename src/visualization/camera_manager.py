import time
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

import viser
from viser import transforms as tf
from viser.extras import ViserUrdf

try:
    from typing import Tuple
except ImportError:
    from builtins import tuple as Tuple


@dataclass
class CameraConfig:
    """相机配置类，管理相机相关参数"""

    joint_name: str = "J_head_yaw"
    fov: float = 1.0  # ~57 degrees
    aspect: float = 1.33  # 4:3 aspect ratio
    height: int = 480
    width: int = 640
    offset_transform: Optional[np.ndarray] = None  # 额外的坐标变换


class CameraManager:
    """高级相机管理类，支持URDF联动和动态更新"""

    def __init__(
        self,
        server: viser.ViserServer,
        viser_urdf: ViserUrdf,
        config: CameraConfig = CameraConfig(),
    ):
        self.server = server
        self.viser_urdf = viser_urdf
        self.config = config
        self.update_callbacks: list[Callable] = []
        self.camera_handle = self._create_camera_frustum()
        self.client_handle = None  # 存储连接的客户端

        # 添加窗口名称
        self.window_name = "Camera View"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.config.width, self.config.height)

        # 设置客户端连接回调
        self.server.on_client_connect(self._handle_client_connect)
        self.server.on_client_disconnect(self._handle_client_disconnect)

    def _handle_client_connect(self, client_id: int, client_handle: viser.ClientHandle):
        """处理客户端连接"""
        print(f"Client {client_id} connected")
        if self.client_handle is None:
            self.client_handle = client_handle

    def _handle_client_disconnect(self, client_id: int):
        """处理客户端断开连接"""
        print(f"Client {client_id} disconnected")
        # 重置客户端句柄
        self.client_handle = None
        # 尝试获取新的客户端
        clients = self.server.get_clients()
        if clients:
            self.client_handle = next(iter(clients.values()))

    def _create_camera_frustum(self) -> viser.CameraFrustumHandle:
        """创建相机模型"""
        cam_pos, cam_quat = self._get_joint_camera_pose()

        camera_handle = self.server.scene.add_camera_frustum(
            name="/camera",
            fov=self.config.fov,
            aspect=self.config.aspect,
            position=cam_pos,
            wxyz=cam_quat,
            visible=False,
            scale=0.3,  # 添加适当的缩放
        )

        return camera_handle

    def _render_camera_view(self):
        """渲染相机视角并使用OpenCV显示"""
        try:
            # 检查是否有可用的客户端句柄
            if self.client_handle is None:
                # 如果没有存储的客户端句柄，尝试获取一个
                clients = self.server.get_clients()
                if not clients:
                    return
                self.client_handle = next(iter(clients.values()))
                print(f"client_id: {self.client_handle.client_id}")

            # 确保client_handle仍然有效
            if self.client_handle not in self.server.get_clients().values():
                self.client_handle = next(iter(self.server.get_clients().values()))

            # start = time.time()
            # 获取渲染结果
            rendered = self.client_handle.get_render(
                height=self.config.height,
                width=self.config.width,
                wxyz=self.camera_handle.wxyz,
                position=self.camera_handle.position,
                fov=self.config.fov,
            )

            # end = time.time()
            # print(f"Time used: {end - start} s")

            # 将RGB格式转换为BGR格式（OpenCV使用BGR）
            bgr_image = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

            # 显示图像
            cv2.imshow(self.window_name, bgr_image)
            cv2.waitKey(1)  # 添加短暂的延迟，允许图像刷新

        except Exception as e:
            print(f"Error rendering camera view: {e}")
            pass

    def _get_joint_camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取关节在世界坐标系下的位置和四元数"""
        if self.config.joint_name not in self.viser_urdf._urdf.joint_map:
            raise ValueError(f"Joint '{self.config.joint_name}' not found in URDF.")

        joint_index = list(self.viser_urdf._urdf.joint_map.keys()).index(
            self.config.joint_name
        )
        frame_handle = self.viser_urdf._joint_frames[joint_index]

        # 应用额外的坐标变换（如果有）
        if self.config.offset_transform is not None:
            # 创建变换矩阵
            world_T_joint_pos = frame_handle.position
            world_T_joint_rot = R.from_quat(frame_handle.wxyz)

            # 应用额外变换
            offset_rot = R.from_matrix(self.config.offset_transform[:3, :3])
            offset_trans = self.config.offset_transform[:3, 3]

            # 组合旋转和平移
            combined_rot = world_T_joint_rot * offset_rot
            combined_trans = world_T_joint_pos + world_T_joint_rot.apply(offset_trans)

            return np.array(combined_trans), np.array(combined_rot.as_quat())

        return frame_handle.position, frame_handle.wxyz

    def update_camera_pose(self):
        """根据URDF关节状态更新相机位姿"""
        try:
            cam_pos, cam_quat = self._get_joint_camera_pose()

            # 更新相机模型位姿
            self.camera_handle.position = cam_pos
            self.camera_handle.wxyz = cam_quat

            # 执行额外的回调函数
            for callback in self.update_callbacks:
                callback(cam_pos, cam_quat)

        except Exception as e:
            print(f"Error updating camera pose: {e}")
            pass

    def add_update_callback(self, callback: Callable):
        """添加相机更新的回调函数"""
        self.update_callbacks.append(callback)

    def __del__(self):
        """析构函数，确保正确关闭OpenCV窗口"""
        cv2.destroyWindow(self.window_name)
