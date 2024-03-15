# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/data/scratch/wangmj/interbotix/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/xs_robot')

from arm import InterbotixManipulatorXS
#from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np
import time
from analytical_ik import AnalyticInverseKinematics as AIK
from scipy.spatial.transform import Rotation as R

# This script makes the end-effector go to a specific pose only possible with a 6dof arm using a transformation matrix
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s'
# Then change to this directory and type 'python ee_pose_matrix_control.py  # python3 bartender.py if using ROS Noetic'

def main():
    # correct joint positions: [ 0.00000000e+00 -3.50647018e-01  9.67456621e-01 -6.16809603e-01 -7.93591507e-06]
    T_sd = np.identity(4)
    T_sd[0,3] = 0.3
    T_sd[1,3] = 0
    T_sd[2,3] = 0.2

    rotation = R.from_matrix(T_sd[:-1, :-1])
    quat = rotation.as_quat()
    
#     T_sd=np.array([[-0.178,  0.247, -0.953,  0.427],
#  [ 0.103,  0.967,  0.231, -0.032],
#  [ 0.979, -0.057, -0.198,  0.324],
#  [ 0.,     0.,     0.,     1.   ]])


    bot = InterbotixManipulatorXS("wx200", "arm", "gripper")
    aik = AIK() 
    #print('current pose', np.array(bot.core.joint_states.position))
    bot.arm.go_to_home_pose()

    print('start set ee')
    start = time.time()
    theta = aik.get_ik(np.array(bot.core.joint_states.position)[:5], 0.3, 0., 0.2, quat[0], quat[1], quat[2], quat[3])
    print('delta time= ', time.time()-start)
    #time.sleep(1)
    #theta, success = bot.arm.set_ee_pose_matrix(T_sd)
    print(theta)
    #time.sleep(1)
    success=bot.arm.set_joint_positions(theta)
    time.sleep(2)
    print(success)
    #bot.arm.go_to_home_pose()
    print('done')
    bot.arm.go_to_sleep_pose()

if __name__=='__main__':
    main()
