# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/data/scratch/wangmj/interbotix/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/xs_robot')

from arm import InterbotixManipulatorXS

#from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np

import time, argparse, threading, tqdm

from queue import Queue
import numpy as np
from IPython import embed
from scipy.spatial.transform import Rotation as R

from mocap.utils.mp_utils import hand_tracker
import trakstar_utils as tkut
import teleop_utils as tlut


parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true")
parser.add_argument("--cons_z", action="store_true")
parser.add_argument("--real", action="store_true", help="teleop on on the real environment as well")
parser.add_argument("--modeon", action="store_true", help="if you are only running this script, then we want to start with modes on. DO NOT CALL THIS FROM teleop.py")
parser.add_argument("--testsim", action="store_true", help="if you want to just test out the different joints (except the arms) in simulation.")
parser.add_argument("--vision", action="store_true", help="Use vision mocap instead of trakstar")
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

def init_poses(robot, trans, rots, vision):

    ef_pos_init, ef_rot_init_mat = get_pose(robot)
    
    ef_rot_init = R.from_matrix(ef_rot_init_mat)
    ms_pos_init, ms_rot_init = get_tf(trans, rots, vision)

    offset = ef_pos_init - ms_pos_init
    roffset = ms_rot_init.inv() * ef_rot_init
    
    return offset, roffset

global get_pose
def get_pose(robot):
    pos = robot.get_ee_pose()
    trans = pos[0:3, 3]
    rot = pos[0:3, 0:3]
    return trans, rot

global get_tf   
def get_tf(trans, rots, vision):

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
    else:
        ms_pos = axis_correction.apply(trans)
        ms_rot = axis_correction * rots

        ms_pos[0] = -ms_pos[0]
        ms_pos[1] = ms_pos[1]

        ms_pos = quat_90.apply(ms_pos)
        #print(ms_pos)
        #ms_rot = R.from_euler('y', 180, degrees=True) * ms_rot
        ms_rot = R.from_euler('z', 180, degrees=True) * ms_rot


    return ms_pos, ms_rot

def get_cmd_q(robot, trans, rots, vision, offset, roffset):
    ms_pos, ms_rot = get_tf(trans, rots, vision)

    ms_pos += offset
    #print('MS_POS           ', ms_pos)
    ms_rot *= roffset

    T_sd = np.identity(4)

    T_sd[0:3, 0:3] = ms_rot.as_matrix()
    T_sd[0:3, 3] = ms_pos

    #print(T_sd)
    thetalist,  _ =robot.set_ee_pose_matrix(T_sd, blocking=False)
    robot.set_joint_positions(thetalist, blocking=False)
    
    pass

def main():
    np.set_printoptions(precision=3, suppress=True)

    bot = InterbotixManipulatorXS("wx200", "arm", "gripper")
    bot.arm.go_to_home_pose()

    """
    Check which kind of teleop mechanism to use
    """
    if args.vision:
        q = Queue()
        tracker=hand_tracker(bimanual=args.bimanual)
        thread = threading.Thread(target=vision_mocap, args=(tracker,q))
        thread.start()

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
    while True:
        if args.vision:
            tr_tuple = q.get()
            q.task_done()
            trans, rots = tr_tuple[0], tr_tuple[1]

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
        
        if not initialized:
            initialized = True
            offset, roffset = init_poses(bot.arm, trans, rots, args.vision)
            get_cmd_q(bot.arm, trans, rots, args.vision, offset, roffset)
            print('OFFSET:   ',offset)
        else:
            get_cmd_q(bot.arm, trans, rots, args.vision, offset, roffset)
    thread.join()

if __name__=='__main__':
    main()