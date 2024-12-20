# robot_manipulation_viser_pybullet_demo.py

import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

sys.path.append(str(Path(__file__).parent.parent))

from src.config.robot_config import RobotConfig
from src.config.camera_config import CameraConfig
from src.simulation.robot_manipulator import RobotManipulator
from src.simulation.robot_manipulator import GraspingTask
from src.simulation.pybullet_env import PyBulletEnvironment
from src.visualization.viser_env import ViserEnvironment
from src.visualization.robot_visualization import RobotVisualization
from src.visualization.camera_manager import CameraManager
from src.synchronization.state_sync import StateSynchronizer
from src.data_loading.gaussian_loader import GaussianSplatLoader
from src.data_loading.mesh_loader import MeshLoader
from src.utils.path_utils import get_corresponding_files


def main():
    # Configuration
    assets_path = Path("io_eai_simulation_assets/assets")
    urdf_path = assets_path / "urdf/AzureLoong_sr.urdf"
    # urdf_path = Path(
    #     "/home/io003/work/io_public/io_teleop_robot_descriptions/OpenLoong/azureloong_description/urdf/AzureLoong_sr.urdf"
    # )

    # Create camera configuration with appropriate transform
    camera_config = CameraConfig(
        joint_name="J_head_yaw",
        offset_transform=create_camera_transform(),
        fov=1.0,
        aspect=1.33,
        height=480,
        width=640,
    )

    # Create robot configuration
    robot_config = RobotConfig(
        urdf_path=urdf_path,
        camera_config=camera_config,
        initial_position=(0.0, 0.0, 1.15),
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        use_fixed_base=True,
    )

    # Initialize Viser server and environments
    viser_env = ViserEnvironment(robot_config)
    pybullet_env = PyBulletEnvironment(robot_config)
    robot_manipulator = RobotManipulator(pybullet_env.robot_id)

    # Initialize robot visualizer and camera manager
    robot_viz = RobotVisualization(viser_env.server, robot_config)
    camera_manager = CameraManager(
        viser_env.server, robot_viz.viser_urdf, config=camera_config
    )

    # Initialize synchronizer
    synchronizer = StateSynchronizer(pybullet_env, viser_env, robot_viz, camera_manager)

    # Load objects
    ply_dir = assets_path / "robot_demo/gsply"
    obj_dir = assets_path / "robot_demo/mesh"
    file_pairs = get_corresponding_files(ply_dir, obj_dir)

    # Load scene
    splat_path = assets_path / "robot_demo/io_lobby.splat"
    scene_data = GaussianSplatLoader.load_splat_file(splat_path, center=False)

    viser_env.add_gaussian_splat(
        "io_lobby",
        scene_data,
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
    )

    # Load meshes in both environments
    mesh_shapes = MeshLoader.load_objects_from_directory(obj_dir)

    for name, (ply_path, obj_path) in file_pairs.items():
        # Load gaussian splat data
        splat_data = GaussianSplatLoader.load_ply_file(ply_path, center=False)

        if name == "apple":
            # Add to both environments
            pybullet_env.add_object(
                name,
                *mesh_shapes[name],
                position=(0.5, 0.0, 0.95),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )

            viser_env.add_gaussian_splat(
                name,
                splat_data,
                position=(0.5, 0.0, 0.95),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

        if name == "fenta":
            # Add to both environments
            pybullet_env.add_object(
                name,
                *mesh_shapes[name],
                position=(0.6, -0.2, 1.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )

            viser_env.add_gaussian_splat(
                name,
                splat_data,
                position=(0.6, -0.2, 1.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

        if name == "orange":
            # Add to both environments
            pybullet_env.add_object(
                name,
                *mesh_shapes[name],
                position=(0.68, 0.14, 0.95),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )

            viser_env.add_gaussian_splat(
                name,
                splat_data,
                position=(0.68, 0.14, 0.95),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

        if name == "bread":
            # Add to both environments
            pybullet_env.add_object(
                name,
                *mesh_shapes[name],
                position=(0.72, 0.33, 0.95),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )

            viser_env.add_gaussian_splat(
                name,
                splat_data,
                position=(0.72, 0.33, 0.95),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

        if name == "banana":
            # Add to both environments
            pybullet_env.add_object(
                name,
                *mesh_shapes[name],
                position=(0.8, 0.0, 0.95),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )

            viser_env.add_gaussian_splat(
                name,
                splat_data,
                position=(0.8, 0.0, 0.95),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

        if name == "table":
            # Add to both environments
            pybullet_env.add_object(
                name,
                *mesh_shapes[name],
                position=(0.75, 0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
                fixed=True,
            )

            viser_env.add_gaussian_splat(
                name,
                splat_data,
                position=(0.75, 0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

    # 创建抓取任务
    grasping_task = GraspingTask(robot_manipulator)

    # 设置目标物体
    target_object_id = pybullet_env.objects["apple"]
    grasping_task.set_target_object(target_object_id)

    # 执行抓取序列
    place_position = [0.1, -1.0, 0.8]  # 放置位置
    grasping_task.execute_grasp_sequence(place_position)

    # Add reset functionality
    def reset_simulation(_):
        pybullet_env.reset()
        robot_viz.reset_state()
        camera_manager.update_camera_pose()

    reset_button = viser_env.server.gui.add_button("Reset Simulation")
    reset_button.on_click(reset_simulation)

    # Main loop
    try:
        while True:
            # Step physics
            pybullet_env.step()

            # Sync states
            synchronizer.sync_all()

            # Render camera view
            camera_manager._render_camera_view()

            # Control rate
            time.sleep(0.02)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


def create_camera_transform() -> np.ndarray:
    """Create camera offset transform matrix"""
    rotation = R.from_euler("xyz", [np.pi / 2, 0, np.pi / 2]).as_matrix()
    translation = np.array([0.25, 0.0, -0.61])

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


if __name__ == "__main__":
    main()
