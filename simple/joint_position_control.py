import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/data/scratch/wangmj/interbotix/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/xs_robot')

from arm import InterbotixManipulatorXS
import time

# This script commands some arbitrary positions to the arm joints
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s'
# Then change to this directory and type 'python joint_position_control.py  # python3 bartender.py if using ROS Noetic'
def torque_on(bot):
    bot.robot_torque_enable("group", "arm", True)
    bot.robot_torque_enable("single", "gripper", True)
def setup_bot(bot):
    bot.robot_reboot_motors("single", "gripper", True)
    bot.robot_set_operating_modes("group", "arm", "position")
    bot.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_on(bot)

def reset_bot(bot):
    bot.robot_set_operating_modes("group", "all", "time")

def main():
    joint_positions = [[1.0, 0.5 , 0.5, 0, -0.5,], [0.9, 0.5 , 0.5, 0, -0.5,], [1.0, 0.5 , 0.5, 0, -0.5,]]
    #joint_positions = [-1.076473325516797, -1.5066153795993784, -0.7010014154374055, 2.2195796979494586, -0.3917391914390844]

    bot = InterbotixManipulatorXS("wx200", "arm", "gripper")
    setup_bot(bot.core)
    bot.arm.go_to_home_pose()
    success=bot.arm.set_joint_positions([1.0, 0.5 , 0.5, 0, -0.5,])
    print(success)
    time.sleep(3)
    for i in range(0, 10):
        jp = [0.9-i*0.1, 0.5 , 0.5, 0, -0.5,]
        success=bot.arm.set_joint_positions(jp)
        print(success)
        #time.sleep(2)
        time.sleep(0.1)
    time.sleep(2)
    bot.arm.go_to_sleep_pose()
    reset_bot(bot.core)

if __name__=='__main__':
    main()
