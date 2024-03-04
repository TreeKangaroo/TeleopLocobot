# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/data/scratch/wangmj/interbotix/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/xs_robot')

from arm import InterbotixManipulatorXS

#from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np

import time, argparse, threading, tqdm, timeit

from queue import Queue
import numpy as np
from IPython import embed
from scipy.spatial.transform import Rotation as R

from mocap.utils.mp_utils import hand_tracker
import trakstar_utils as tkut
import teleop_utils as tlut
import pybullet as p



def wrap_theta_list(bot, theta_list):
    """
    Wrap an array of joint commands to [-pi, pi) and between the joint limits.

    :param theta_list: array of floats to wrap
    :return: array of floats wrapped between [-pi, pi)
    """
    REV = 2*np.pi
    theta_list = (theta_list + np.pi) % REV - np.pi
    for x in range(0,len(theta_list)):
        if round(theta_list[x], 3) < round(bot.arm.group_info.joint_lower_limits[x], 3):
            theta_list[x] += REV
        elif round(theta_list[x], 3) > round(bot.arm.group_info.joint_upper_limits[x], 3):
            theta_list[x] -= REV
    return theta_list


def main():
    with open('jointsp.npy', 'rb') as f:
        a = np.load(f)

    physicsClient = p.connect(p.GUI)
    p.setTimeStep( 1.0 / 1000)
    p.setGravity(0,0,-9.81)
    p.setRealTimeSimulation(1)
    #bot_chain = Chain.from_urdf_file('/data/scratch/wangmj/interbotix/src/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/wx200.urdf.xacro',
                                       # base_elements=['world'])
    robotId = p.loadURDF('/data/scratch/wangmj/interbotix/src/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/wx200.urdf',
                  useFixedBase=True)

    #bot_chain = Chain.from_urdf_file('./locobot/assets/locobot_description/locobot.urdf',
                                       # base_elements=['arm_base_link'])
    
    bot = InterbotixManipulatorXS("wx200", "arm", "gripper")
    bot.arm.go_to_home_pose()

    for row in a:
        #thetalist,  _ =bot.arm.set_ee_pose_matrix(row, blocking=False)
        #print('theta    ',thetalist)
        #ms_pos = row[0:3, 3].flatten()
        #ms_rot= R.from_matrix(row[0:3, 0:3])
        #ik_res = p.calculateInverseKinematics(robotId, 4, ms_pos, ms_rot.as_quat(), maxNumIterations=100,residualThreshold=.01)
        #ik = wrap_theta_list(bot, np.array(ik_res[0:5]))
        #print('pybullet    ', ik)
        bot.arm.set_joint_positions(row, blocking=False)
        #bot.arm.set_joint_positions(row, blocking=False)
        time.sleep(0.02)

    print('DONE')
if __name__=='__main__':
    main()