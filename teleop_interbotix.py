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
from avp_stream import VisionProStreamer


from mocap.utils.mp_utils import hand_tracker
import trakstar_utils as tkut
import teleop_utils as tlut
import pybullet as p
from oculus_reader.oculus_reader.reader import OculusReader
from simple.analytical_ik import AnalyticInverseKinematics as AIK

parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true")
parser.add_argument("--cons_z", action="store_true")
parser.add_argument("--real", action="store_true", help="teleop on on the real environment as well")
parser.add_argument("--modeon", action="store_true", help="if you are only running this script, then we want to start with modes on. DO NOT CALL THIS FROM teleop.py")
parser.add_argument("--testsim", action="store_true", help="if you want to just test out the different joints (except the arms) in simulation.")
parser.add_argument("--vision", action="store_true", help="Use vision mocap instead of trakstar")
parser.add_argument("--oculus", action="store_true", help="Use oculus joystick instead of trakstar")
parser.add_argument("--visionpro", action="store_true", help="Use oculus joystick instead of trakstar")
parser.add_argument("--trakstar", action="store_true", help="Use trakstar")
parser.add_argument("--bimanual", action="store_true", help="Use both hands to move both of the robot's arms")
args = parser.parse_args()

def vision_mocap(ht, qu):
    while True:
        ht.update()
        ht.solvepose()
        ht.visualize()

        if args.bimanual:
            trans_r, rots_in_r = ht.translation_vector_r, ht.rotation_matrix_r
            trans_l, rots_in_l = ht.translation_vector_l, ht.rotation_matrix_l

            rots_r = R.from_matrix(rots_in_r)
            rots_l = R.from_matrix(rots_in_l)

            qu.put((trans_r, rots_r, trans_l, rots_l))
        else:
            trans, rots_in = ht.translation_vector, ht.rotation_matrix
            rots = R.from_matrix(rots_in)
            qu.put((trans,rots, -1, -1))

    # After the loop release the cap object 
    ht.vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 

def init_poses(robot, trans, rots, vision, oculus):

    ef_pos_init, ef_rot_init_mat = get_pose(robot)
    
    ef_rot_init = R.from_matrix(ef_rot_init_mat)
    ms_pos_init, ms_rot_init = get_tf(trans, rots, vision, oculus)

    offset = ef_pos_init - ms_pos_init
    roffset = ms_rot_init.inv() * ef_rot_init
    
    return offset, roffset
DT=0.02

global get_pose
def get_pose(robot):
    pos = robot.get_ee_pose()
    trans = pos[0:3, 3]
    rot = pos[0:3, 0:3]
    return trans, rot

global get_tf   
def get_tf(trans, rots, vision=False, oculus = False):

    quat_90 = R.from_quat([ 0, 0, 0.7071081, 0.7071055 ])
    idk = R.from_euler('z',-90, degrees=True)
    axis_correction = R.from_euler('x', 180, degrees=True)

    if vision:
        ms_pos_uncorrected = axis_correction.apply(trans.flatten())
        
        #ms_rot = self.axis_correction * rots
        ms_pos=np.zeros((3))
        ms_pos[2]=ms_pos_uncorrected[1]
        ms_pos[1]=ms_pos_uncorrected[0]
        ms_pos[0]=ms_pos_uncorrected[2]
        
        #ms_pos = self.quat_90.apply(ms_pos)
        ms_rot = rots
    if oculus:
        #fix translation
        ms_pos_uncorrected = trans.flatten()
        ms_pos=np.zeros((3))
        ms_pos[0]=ms_pos_uncorrected[2]
        ms_pos[1]= ms_pos_uncorrected[0]
        ms_pos[2]= ms_pos_uncorrected[1]

        #fix rotation
        euler = rots.as_euler('xyz', degrees=True)
        ms_rot = np.zeros((3))
        ms_rot[0]=euler[1]
        ms_rot[1]=euler[0]
        ms_rot[2]=-euler[2]
        ms_rot = R.from_euler('zyx', ms_rot, degrees=True)
    else:
        """
        ms_pos = axis_correction.apply(trans)
        ms_rot = axis_correction * rots

        ms_pos[0] = -ms_pos[0]
        ms_pos[1] = ms_pos[1]

        ms_pos = quat_90.apply(ms_pos)
        #print(ms_pos)
        #ms_rot = R.from_euler('y', 180, degrees=True) * ms_rot
        ms_rot = R.from_euler('z', 180, degrees=True) * ms_rot
        """
        ms_pos = axis_correction.apply(trans)
        ms_rot = axis_correction * rots

        ms_pos[0] = -ms_pos[0]
        ms_pos[1] = -ms_pos[1]

        ms_pos = quat_90.apply(ms_pos)
        ms_rot = idk * ms_rot

    return ms_pos, ms_rot
def ang_in_mpi_ppi(angle):
    angle = (angle + np.pi) % (2 * np.pi) - np.pi

    return angle

ll= [-3.141582727432251, -1.884955644607544, -1.884955644607544, -1.7453292608261108, -3.141582727432251]
ul = [3.141582727432251, 1.972222089767456, 1.623156189918518, 2.1467549800872803, 3.141582727432251]
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
rest = [0,0,0,0,0]
def get_cmd_q(robot, aik, robotId, trans, rots, vision, oculus, offset, roffset):
    ms_pos, ms_rot = get_tf(trans, rots, vision, oculus)

    ms_pos += offset
    print
    #print('MS_POS           ', ms_pos)
    ms_rot *= roffset
    ms_rot_quat = ms_rot.as_quat()

    #print(T_sd)
    def wrap_theta_list(bot, theta_list):
        """
        Wrap an array of joint commands to [-pi, pi) and between the joint limits.

        :param theta_list: array of floats to wrap
        :return: array of floats wrapped between [-pi, pi)
        """
        REV = 2*np.pi
        theta_list = (theta_list + np.pi) % REV - np.pi
        for x in range(0,len(theta_list)):
            if round(theta_list[x], 3) < round(bot.group_info.joint_lower_limits[x], 3):
                theta_list[x] += REV
            elif round(theta_list[x], 3) > round(bot.group_info.joint_upper_limits[x], 3):
                theta_list[x] -= REV
        return theta_list

    # start = timeit.default_timer()
    
    # ik_res = p.calculateInverseKinematics(robotId, 4, ms_pos, ms_rot.as_quat()) #lowerLimits = ll, upperLimits = ul, jointRanges = jr, restPoses=rest, maxNumIterations=100,residualThreshold=.01)
    # ik = [ang_in_mpi_ppi(ang) for ang in ik_res[0:5]]
    # print(ik)
    # p.setJointMotorControlArray(robotId, [0,1,2,3,4], p.POSITION_CONTROL, ik_res[0:5])
    # success=robot.set_joint_positions(ik)
    # print(success)
    # print("The difference of time is :", 
    #           timeit.default_timer() - start)

    # p.stepSimulation()
    # time.sleep(0.02)
    # return ik
    ik_res = p.calculateInverseKinematics(robotId, 4, ms_pos, ms_rot_quat)
    p.setJointMotorControlArray(robotId, [0,1,2,3,4], p.POSITION_CONTROL, ik_res[0:5])

    start = timeit.default_timer()

    ik = aik.get_ik(np.array(robot.core.joint_states.position)[:5], ms_pos[0], ms_pos[1], ms_pos[2], 
                    ms_rot_quat[0], ms_rot_quat[1], ms_rot_quat[2], ms_rot_quat[3])
    dif= timeit.default_timer() - start
    if ik:
        #print('success')
        success=robot.arm.set_joint_positions(ik, blocking = False)

    
    p.stepSimulation()
    time.sleep(0.02)
    return dif
def torque_on(bot):
    bot.robot_torque_enable("group", "arm", True)
    bot.robot_torque_enable("single", "gripper", True)
def setup_bot(bot):
    bot.robot_reboot_motors("single", "gripper", True)
    bot.robot_set_operating_modes("group", "arm", "position")
    bot.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_on(bot)

def main():

    np.set_printoptions(precision=3, suppress=True)

    bot = InterbotixManipulatorXS("wx200", "arm", "gripper")
    aik = AIK() 
    #bot.dxl.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 800)
    #bot.dxl.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)
    setup_bot(bot.core)
    bot.arm.go_to_sleep_pose()
    
    physicsClient = p.connect(p.GUI)
    p.setTimeStep( 1.0 / 100)
    p.setGravity(0,0,-9.81)
    #p.setRealTimeSimulation(1)
   
    robotId = p.loadURDF('/data/scratch/wangmj/interbotix/src/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/wx200.urdf',
                  useFixedBase=True)
    """
    Check which kind of teleop mechanism to use
    """
    if args.vision:
        q = Queue()
        tracker=hand_tracker(bimanual=args.bimanual)
        thread = threading.Thread(target=vision_mocap, args=(tracker,q))
        thread.start()
    elif args.visionpro:
        avp_ip = "10.29.230.57"   # example IP 
        s = VisionProStreamer(ip = avp_ip, record = True)
    elif args.oculus:
        oculus_reader = OculusReader()

    else:
        ts_node = tkut.ts_node1

    if args.trakstar:
        pbar = tqdm.tqdm(range(100), desc="Pinging trakstar...")

        for i in pbar:
            ts_node.socket.send_string("sup")
            ts_node.get_data()

    ms_pos, ms_rot_mat = get_pose(bot.arm)
    ms_rot = R.from_matrix(ms_rot_mat)
    """
    Main loop; has no safety constraints added
    """
    initialized = False
    oculus_ready = False

    arrayf = np.zeros((500, 5))
    arrayik = np.zeros((500,4,4))
    array_times = np.zeros((500))
    count = 0
    
    while True:
        if args.vision:
            tr_tuple = q.get()
            q.task_done()
            trans, rots = tr_tuple[0], tr_tuple[1]
        elif args.visionpro:
            r = s.latest
            trans = r['head'][0,:-1, 3]
            rots = R.from_matrix(r['head'][0,:-1, :-1])

        elif args.oculus:
            matrix, _ = oculus_reader.get_transformations_and_buttons()
            if 'r' in matrix:
                trans = matrix['r'][:-1, 3]
                rots = R.from_matrix(matrix['r'][:-1, :-1])
                oculus_ready=True

            else:
                trans=np.array([0,0,0])
                rots=R.from_matrix(np.array([[1,0,0], [0,1,0], [0,0,1]]))

        else:
            ts_node.socket.send_string("sup")
            trans_in, rots_in = ts_node.get_data()
            trans, rots=trans_in[3], rots_in[3]
            #print('Trans:    ', trans)
            


            
            pos_thumb = trans_in[0]
            pos_index = trans_in[1]
            cdist_thumb_index = np.linalg.norm(pos_thumb - pos_index)
            if cdist_thumb_index < 0.03:
                bot.gripper.grasp()
            else:
                bot.gripper.release()
            
        ms_pos, ms_rot_mat = get_pose(bot.arm)
        ms_rot = R.from_matrix(ms_rot_mat)
        
        if not initialized and oculus_ready:
            initialized = True
            offset, roffset = init_poses(bot.arm, trans, rots, args.vision, args.oculus)
            get_cmd_q(bot, aik, robotId, trans, rots, args.vision, args.oculus, offset, roffset)
            
            print('OFFSET:   ',offset)
        elif initialized:
            time=get_cmd_q(bot, aik, robotId, trans, rots, args.vision, args.oculus, offset, roffset)
            
            if count<500:
                #arrayf[count, :] = ik
                #arrayik[count, :, :] = T_sd
                array_times[count] = time
            if count ==500:
                print('SAVING')
                with open('iktimes.npy', 'wb') as f:
                    np.save(f, array_times)
                #with open('jointsp.npy', 'wb') as f:
                    #np.save(f, arrayf)
                #with open('ik.npy', 'wb') as g:
                    #np.save(g, arrayik)
            count+=1
    time.sleep(0.02)        

if __name__=='__main__':
    main()