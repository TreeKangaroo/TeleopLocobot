from dataclasses import dataclass

import numpy as np
import pybullet as p
from airobot.utils.common import to_quat
from airobot.utils.common import to_rot_mat
from scipy.spatial.transform import Rotation as R


@dataclass
class Locobot:
    bot: int

    def __post_init__(self):
        self.count=0
        self.wheel_joints = [1, 2]  # Left and right wheels
        self.wheel_default_vel = 10
        self.arm_joints = [13, 14, 15, 16, 17]  # Arm joints
        self.gripper_joints = [18, 19]  # Left and right
        self.ee_link = self.arm_joints[-2]  # Link to which ee is attached
        self.camera_link = 24
        self.camera_motor_joints = [22, 23]

        # some configurations for the joints
        self.homej = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # default config for arm
        self.ee_half_width = 0.025  # half-width of gripper
        self.close_ee_config = np.array([-1e-3, 1e-3])
        self.open_ee_config = self.ee_half_width * np.array([-1, 1])

        self.arm_kp = np.array([0.08] * 5)
        self.arm_kd = np.array([0.3] * 5)
        self.camera_motor_kp = np.array([0.03] * 2)
        self.camera_motor_kd = np.array([0.3] * 2)
        self.arm_jnt_max_force = 1
        self.gripper_jnt_max_force = 1
        self.wheel_jnt_max_force = 20
        self.gripper_kp = np.array([0.08] * 2)
        self.gripper_kd = np.array([0.3] * 2)
        self.num_joints = p.getNumJoints(self.bot)
        self.joints_info = [p.getJointInfo(self.bot, i) for i in range(self.num_joints)]

        camera_offset1 = R.from_rotvec(np.array([0, 1, 0]) * np.pi / 2)
        camera_offset2 = R.from_rotvec(-np.array([0, 0, 1]) * np.pi / 2)
        self.cam_offset = np.dot(camera_offset1.as_matrix(), camera_offset2.as_matrix())
        self.initialized = False

    def reset(self):
        
        for i in range(len(self.arm_joints)):
            p.resetJointState(self.bot, self.arm_joints[i], self.homej[i])
        self.open_gripper()
        
    def get_jpos(self, joints):
        states = p.getJointStates(self.bot, joints)
        pos = [state[0] for state in states]
        return np.array(pos)

    def get_arm_jpos(self):
        return self.get_jpos(self.arm_joints)

    def get_gripper_jpos(self):
        return self.get_jpos(self.gripper_joints)

    def get_wheel_jpos(self):
        return self.get_jpos(self.wheel_joints)

    def get_base_pose(self):
        pos, ori = p.getBasePositionAndOrientation(self.bot)
        return np.array(pos), np.array(ori)

    def get_locobot_camera_pose(self):
        info = p.getLinkState(self.bot, self.camera_link)
        pos = info[4]
        quat = info[5]
        pos = np.array(pos)
        quat = np.array(quat)
        rot_mat = to_rot_mat(quat)
        offset_rot_mat = np.dot(rot_mat, self.cam_offset)
        offset_quat = to_quat(offset_rot_mat)
        return pos, offset_quat

    def set_locobot_camera_pan_tilt(self, pan, tilt, ignore_physics=True):
        if not ignore_physics:
            p.setJointMotorControlArray(
                bodyIndex=self.bot,
                jointIndices=self.camera_motor_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[pan, tilt],
                forces=self.arm_jnt_max_force * np.ones(2),
                positionGains=self.camera_motor_kp,
                velocityGains=self.camera_motor_kp)
        else:
            p.resetJointState(self.bot,
                              self.camera_motor_joints[0],
                              targetValue=pan,
                              targetVelocity=0)
            p.resetJointState(self.bot,
                              self.camera_motor_joints[1],
                              targetValue=tilt,
                              targetVelocity=0)

    def set_base_vel(self, vels):
        p.setJointMotorControlArray(bodyIndex=self.bot,
                                    jointIndices=self.wheel_joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=vels,
                                    forces=self.wheel_jnt_max_force * np.ones(2))

    def rotate_to_left(self):
        self.set_base_vel([-self.wheel_default_vel, self.wheel_default_vel])

    def rotate_to_right(self):
        self.set_base_vel([self.wheel_default_vel, -self.wheel_default_vel])

    def base_forward(self):
        self.set_base_vel([self.wheel_default_vel, self.wheel_default_vel])

    def base_backward(self):
        self.set_base_vel([-self.wheel_default_vel, -self.wheel_default_vel])

    def stop_base(self):
        self.set_base_vel([0, 0])

    def set_arm_jpos(self, jpos):
        p.setJointMotorControlArray(
            bodyIndex=self.bot,
            jointIndices=self.arm_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=jpos)
            #forces=self.arm_jnt_max_force * np.ones(len(jpos)),
            #positionGains=self.arm_kp,
            #velocityGains=self.arm_kd)

    def set_gripper_jpos(self, jpos):
        p.setJointMotorControlArray(
            bodyIndex=self.bot,
            jointIndices=self.gripper_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=jpos,
            forces=self.gripper_jnt_max_force * np.ones(len(jpos)),
            positionGains=self.gripper_kp,
            velocityGains=self.gripper_kd)

    def get_ee_pose(self):
        self.count+=1
        pos, quat = p.getLinkState(self.bot, self.ee_link)[-2:]
        #if self.count%100==0:
            #print(pos, quat)
        return np.array(pos), np.array(quat)

    def set_ee_pose(self, position, orientation):
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.bot,
            endEffectorLinkIndex=self.ee_link,
            targetPosition=position,
            targetOrientation=orientation,
            restPoses=self.homej.tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        arm_joints_qs = np.array(joints[2:7])
        self.set_arm_jpos(arm_joints_qs)
        return arm_joints_qs

    def close_gripper(self):
        self.set_gripper_jpos(self.close_ee_config)

    def open_gripper(self):
        self.set_gripper_jpos(self.open_ee_config)

    def init_poses(self, trans, rots, vision):
        self.quat_90 = R.from_quat([ 0, 0, 0.7071081, 0.7071055 ])
        self.idk = R.from_euler('z',-90, degrees=True)
        self.axis_correction = R.from_euler('x', 180, degrees=True)

        ef_pos_init, ef_rot_init_quat = self.get_ee_pose()
        
        ef_rot_init = R.from_quat(ef_rot_init_quat)
        ms_pos_init, ms_rot_init = self.get_tf(trans, rots, vision)

        self.offset = ef_pos_init - ms_pos_init
        self.roffset = ms_rot_init.inv() * ef_rot_init
        
        self.initialized = True
    
    def get_tf(self, trans, rots, vision):

        if vision:
            ms_pos_uncorrected = self.axis_correction.apply(trans.flatten())
            #ms_rot = self.axis_correction * rots
            ms_pos=np.zeros((3))
            ms_pos[2]=ms_pos_uncorrected[1]
            ms_pos[1]=ms_pos_uncorrected[0]
            ms_pos[0]=ms_pos_uncorrected[2]
            
            #ms_pos = self.quat_90.apply(ms_pos)
            ms_rot = rots
        else:
            ms_pos = self.axis_correction.apply(trans)
            ms_rot = self.axis_correction * rots

            ms_pos[0] = -ms_pos[0]
            ms_pos[1] = ms_pos[1]

            ms_pos = self.quat_90.apply(ms_pos)
            ms_rot = self.idk * ms_rot

        return ms_pos, ms_rot

    def get_cmd_q(self, trans, rots, vision):
        ms_pos, ms_rot = self.get_tf(trans, rots, vision)

        ms_pos += self.offset
        ms_rot *= self.roffset

        self.set_ee_pose(ms_pos, ms_rot.as_quat())

        pass
