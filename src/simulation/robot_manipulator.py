import numpy as np
import pybullet as p
from typing import Optional, Tuple, List
import math
import time


class RobotManipulator:
    def __init__(self, robot_id: int):
        self.robot_id = robot_id
        self.control_dt = 1.0 / 500.0
        self.end_effector_index = 11  # 末端执行器关节索引，需要根据URDF调整
        self.arm_joints = list(range(4, 11))  # J_arm_r_01 to J_arm_r_07
        self.num_joints = 7  # 机器人关节数量

        self.ff_joints = list(range(14, 18))  # First finger joints
        self.mf_joints = list(range(19, 23))  # Middle finger joints
        self.rf_joints = list(range(24, 28))  # Ring finger joints
        self.lf_joints = list(range(29, 34))  # Little finger joints
        self.th_joints = list(range(35, 40))  # Thmub finger joints

        # 设置关节限制
        self.lower_limits = []
        self.upper_limits = []
        self.joint_ranges = []

        self.joint_limits = {
            4: [-np.pi * 2, np.pi * 2],  # J_arm_r_01
            5: [-np.pi * 2, np.pi * 2],  # J_arm_r_02
            6: [-np.pi * 2, np.pi * 2],  # J_arm_r_03
            7: [-np.pi * 2, np.pi * 2],  # J_arm_r_04
            8: [-np.pi * 2, np.pi * 2],  # J_arm_r_05
            9: [-np.pi * 2, np.pi * 2],  # J_arm_r_06
            10: [-np.pi * 2, np.pi * 2],  # J_arm_r_07
        }

        # 初始关节位置
        self.initial_positions = {
            3: -0.5,  # J_head_pitch
            4: 0.1956304907798767,  # J_arm_r_01
            5: -0.968996524810791,  # J_arm_r_02
            6: -2.003570795059204,  # J_arm_r_03
            7: 1.6953843832015991,  # J_arm_r_04
            8: 0.6260079741477966,  # J_arm_r_05
            9: -0.044389571994543,  # J_arm_r_06
            10: -0.426590472459793,  # J_arm_r_07
            41: -0.1956304907798767,  # J_arm_l_01
            42: -0.9689006209373474,  # J_arm_l_02
            43: 2.003570795059204,  # J_arm_l_03
            44: 1.6953364610671997,  # J_arm_l_04
            45: -0.6255286335945129,  # J_arm_l_05
            46: -0.045875612646341324,  # J_arm_l_06
            47: 0.4266384243965149,  # J_arm_l_07
        }

        self._setup_robot()

    def _setup_robot(self):
        """初始化机器人配置"""
        # 设置关节阻尼
        for j in range(p.getNumJoints(self.robot_id)):
            p.changeDynamics(self.robot_id, j, linearDamping=0, angularDamping=0)

        # 初始化关节位置
        for joint_id, pos in self.initial_positions.items():
            p.resetJointState(self.robot_id, joint_id, pos)
            time.sleep(0.1)

    def calculate_ik(
        self, target_pos: List[float], target_orn: Optional[List[float]] = None
    ) -> List[float]:
        current_joint_states = [
            p.getJointState(self.robot_id, j)[0] for j in self.arm_joints
        ]

        """计算逆运动学"""
        # if target_orn is None:
        #     target_orn = p.getQuaternionFromEuler([math.pi / 2.0, math.pi / 2.0, 0.0])

        for joint_id in self.arm_joints:
            if joint_id in self.joint_limits:
                self.lower_limits.append(self.joint_limits[joint_id][0])
                self.upper_limits.append(self.joint_limits[joint_id][1])
                self.joint_ranges.append(
                    self.joint_limits[joint_id][1] - self.joint_limits[joint_id][0]
                )

        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_pos,
            target_orn,
            self.lower_limits,
            self.upper_limits,
            self.joint_ranges,
            restPoses=current_joint_states,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        return joint_poses

    def move_to_target(
        self, target_pos: List[float], target_orn: Optional[List[float]] = None
    ):
        """移动到目标位置"""
        joint_poses = self.calculate_ik(target_pos, target_orn)

        # 控制关节运动
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL, joint_poses[i], force=5 * 240.0
            )

    def control_gripper(self, gripper_width):
        """
        Control gripper width
        Args:
            gripper_width: 0.0 for closed, 1.0 for open
        """
        # Map gripper_width (0-1) to joint angles
        finger_angle = 1.57 * (1 - gripper_width)  # Max angle is 1.57 (90 degrees)

        # Control all finger joints
        for finger_joints in [
            self.ff_joints,
            self.mf_joints,
            self.rf_joints,
            self.lf_joints,
            self.th_joints,
        ]:
            # Skip the first joint (lateral movement) and control flexion joints
            for joint_id in finger_joints[1:]:  # FFJ3, FFJ2, FFJ1 for each finger
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=finger_angle,
                    force=2.0,  # Using max force from URDF
                )

    def get_end_effector_pose(self) -> Tuple[List[float], List[float]]:
        """获取末端执行器的位姿"""
        state = p.getLinkState(self.robot_id, self.end_effector_index)
        return state[0], state[1]  # position, orientation


class GraspingTask:
    def __init__(self, robot_manipulator: RobotManipulator):
        self.robot = robot_manipulator
        self.state = 0
        self.target_object_id = None
        self.gripper_height = 0.2

    def set_target_object(self, object_id: int):
        """设置要抓取的目标物体"""
        self.target_object_id = object_id

    def approach_object(self):
        """接近物体"""
        if self.target_object_id is None:
            return

        print("Approching to object!")

        self.robot.control_gripper(1.0)

        # 获取物体位置
        pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
        # 设置抓取位置（在物体上方）
        target_pos = [pos[0], pos[1], pos[2] + 0.05]
        self.robot.move_to_target(target_pos)

    def grasp_object(self):
        """抓取物体"""
        if self.target_object_id is None:
            return

        # 获取物体位置
        pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
        # 移动到抓取位置
        target_pos = [pos[0], pos[1], pos[2]]
        self.robot.move_to_target(target_pos)
        # 闭合夹爪
        time.sleep(0.5)
        self.robot.control_gripper(0.01)

        print("Object grasped!")

    def lift_object(self):
        """提起物体"""
        if self.target_object_id is None:
            return

        current_pos, _ = self.robot.get_end_effector_pose()
        # 向上提起
        target_pos = [current_pos[0], current_pos[1], current_pos[2] + 0.2]
        self.robot.move_to_target(target_pos)

        print("Object lifted!")

    def place_object(self, target_pos: List[float]):
        """放置物体"""
        # 移动到目标位置上方
        self.robot.move_to_target([target_pos[0], target_pos[1], target_pos[2] + 0.2])
        # 降低到放置位置
        self.robot.move_to_target(target_pos)
        # 打开夹爪
        self.robot.control_gripper(0.04)

    def execute_grasp_sequence(self, place_position: List[float]):
        """执行完整的抓取序列"""
        # 接近物体
        self.approach_object()
        p.stepSimulation()

        # 抓取物体
        self.grasp_object()
        p.stepSimulation()

        # 提起物体
        self.lift_object()
        p.stepSimulation()

        # 放置物体
        self.place_object(place_position)
        p.stepSimulation()
