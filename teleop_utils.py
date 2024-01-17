import tqdm, time, math, pybullet as p, numpy as np

from scipy.spatial.transform import Rotation as R
from enum import Enum
#from std_msgs.msg import String
from collections import deque

# original right arm pose (from Branden for a single-arm): [-4.63707667986025, -1.627172132531637, 2.1350348631488245, -0.651104287510254, 1.462552547454834, -0.9959004561053675]
RIGHT_ARM_GOOD_POSE = [-4.606921736394064, -1.574414078389303, 2.0893893241882324, -0.5094082991229456, 1.6109023094177246, 0] # first is the base, last is the wrist.
LEFT_ARM_GOOD_POSE = [4.606921736394064, -1.574414078389303, -2.0893893241882324, -2.6034608682, -1.6109023094177246, 0]

DIST_CONFIG = {
    0: [0.035, 2200] # for Bipasha
}

class MODES(Enum):
    ON = True
    OFF = False
    
operation_mode = MODES.OFF
SWITCH_ITERATIONS = 0

SAFETY = MODES.OFF

def turn_safety_on(on=True):
    global SAFETY
    SAFETY = MODES.ON if on else MODES.OFF

def is_safety_on():
    return SAFETY == MODES.ON

def set_operation_mode(new_mode):
    global operation_mode
    operation_mode = new_mode

def get_operation_mode():
    global operation_mode
    return operation_mode
    
def switch_callback(data, robot=""):
    global SWITCH_ITERATIONS
    mode = get_operation_mode() 

    if mode == MODES.ON and data.data == "pause":
        # print("paused", robot, "!")
        set_operation_mode(MODES.OFF)

    elif mode == MODES.OFF and SAFETY == MODES.OFF and data.data == "resume":
        set_operation_mode(MODES.ON)

        # print("resumed", robot, "!")

    elif mode == MODES.ON and data.data == "terminate":
        # print("shutting down the", robot,"!")
        set_operation_mode(MODES.OFF)

        rospy.signal_shutdown("Termination command received")

    elif data.data == "protective":
        set_operation_mode(MODES.OFF)

        turn_safety_on()

    elif data.data == "safe":
        turn_safety_on(on=False)

    else:
        return

    SWITCH_ITERATIONS +=1
    print("#", SWITCH_ITERATIONS, ": Received message at", robot,":", data.data)

def get_robot_right_joint_q(args, rtde_r):
    # Bringing the robot and the pybullet in the same view.
    # Uncomment / modify cautiously, it can create very weird movements for the robot.
    if args.reset or not args.real:
        # don't remove, this is a safe position.
        robot_right_joint_q = RIGHT_ARM_GOOD_POSE
        robot_left_joint_q = LEFT_ARM_GOOD_POSE
    else:
        robot_right_joint_q = rtde_r.getActualQ()
        robot_left_joint_q = LEFT_ARM_GOOD_POSE # TODO, change and get it from the real robot

    return robot_right_joint_q, robot_left_joint_q

def get_mag_sense_pose(id, listener):
    sensor = "/mag_sensor_{}".format(id)
    world = "/trak_star"
    axis_correction = R.from_euler('x', 180, degrees=True)
    try:
        listener.waitForTransform(sensor, world, rospy.Time(0), rospy.Duration(5))
        (trans,rot) = listener.lookupTransform(world, sensor, rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        trans = [0,0,0]
        rot = [0,0,0,1]
        print("cant find mag sensor")
    trans = np.array(trans)
    rot = R.from_quat(rot)
    return axis_correction.apply(trans), axis_correction * rot

def print_visualizer_camera_settings():
    camera_settings = p.getDebugVisualizerCamera()

    camera_settings_dict = {
        "cameraDistance": camera_settings[10],
        "cameraYaw": camera_settings[8],
        "cameraPitch": camera_settings[9],
        "cameraTargetPosition": camera_settings[11:14]
    }

    print("Camera Settings: ", camera_settings_dict)

def axiscreator():
    x_axis = p.addUserDebugLine(lineFromXYZ = [0, 0, 0],
                                lineToXYZ = [0.1, 0, 0],
                                lineColorRGB = [1, 0, 0])

    y_axis = p.addUserDebugLine(lineFromXYZ = [0, 0, 0],
                                lineToXYZ            = [0, 0.1, 0],
                                lineColorRGB         = [0, 1, 0])

    z_axis = p.addUserDebugLine(lineFromXYZ = [0, 0, 0],
                                lineToXYZ            = [0, 0, 0.1],
                                lineColorRGB         = [0, 0, 1])

    return [x_axis, y_axis, z_axis]

def axiscreator2(quat,trans,prev_ids):
    origin = np.zeros(3) + trans
    end_points = quat.apply(np.eye(3) * .1) + trans
    x_axis = p.addUserDebugLine(lineFromXYZ = origin,
                                lineToXYZ = end_points[0],
                                lineColorRGB = [1, 0, 0],
                                lineWidth = 0.1,
                                replaceItemUniqueId = prev_ids[0] )

    y_axis = p.addUserDebugLine(lineFromXYZ = origin,
                                lineToXYZ            = end_points[1],
                                lineColorRGB         = [0, 1, 0] ,
                                lineWidth            = 0.1,
                                replaceItemUniqueId = prev_ids[1])

    z_axis = p.addUserDebugLine(lineFromXYZ = origin,
                                lineToXYZ            = end_points[2],
                                lineColorRGB         = [0, 0, 1],
                                lineWidth            = 0.1,
                                replaceItemUniqueId = prev_ids[2])

    return [x_axis, y_axis, z_axis]

def safely_unlock(dash, rtde_c, rtde_r):
    set_operation_mode(MODES.OFF)
    turn_safety_on()

    for _ in tqdm.tqdm(range(8), desc="Resuming in", dynamic_ncols=True):
        time.sleep(1)  
    
    input("Need User Input, press any key to continue")
    dash.unlockProtectiveStop()
    for t in range(2):
        try:
            print("#",t,"Trying to reconnect...")

            rtde_c.disconnect()
            rtde_r.disconnect()

            rtde_c.reconnect()
            rtde_r.reconnect()
        except:
            pass

    return deque(maxlen=100) # reset arm_poses_buff

def getrtdeInitPeriod(args, rtde_c):
    if args.real:
        return rtde_c.initPeriod()

    return 0

def rtdeWait(args, rtde_c, t_start):
    if args.real:
        rtde_c.waitPeriod(t_start)

def stop_rtde_c(rtde_c):
    rtde_c.servoStop()
    rtde_c.stopScript()

def on_exception(args, e):
    if args.real:
        stop_rtde_c()
    else:
        pass #code here for simulation, TODO
    print("exeception at the arm:", e," !")

def is_within_torque_limits(torque, torque_limit):
    return np.all(torque[3:] < torque_limit[3:]) and np.all(torque[1:3] > torque_limit[1:3]) and torque[0] < torque_limit[0]

def move_real_arm_to_posiiton(args, rtde_c, ik_res, velocity, acceleration, dt, lookahead_time, gain):
    if args.real:
        rtde_c.servoJ(ik_res, velocity, acceleration, dt, lookahead_time, gain)
    else:
        pass

def triggerProtectivestop(args, rtde_c):
    if args.real:
        rtde_c.triggerProtectiveStop()
    else:
        pass #code here, TODO

def getCurrentTorque(args, rtde_c, torque_limit):
    if args.real:
        current_torque = np.array(rtde_c.getJointTorques())
    else: 
        current_torque = np.zeros_like(torque_limit)
        pass #TODO - add code here. how to measure torque in simulation?

    return current_torque