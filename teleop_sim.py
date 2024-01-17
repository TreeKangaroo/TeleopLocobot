import time, argparse, threading, tqdm

from queue import Queue
import numpy as np
from IPython import embed
from scipy.spatial.transform import Rotation as R

from locobot.sim.base_env import BaseEnv
from locobot.sim.locobot import Locobot
from mocaptest.utils.mp_utils import hand_tracker
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


np.set_printoptions(precision=3, suppress=True)

env = BaseEnv(realtime=True)
env.reset()
bot = env.bot.bot


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

ms_pos, ms_rot_quat = env.bot.get_ee_pose()
ms_rot = R.from_quat(ms_rot_quat)

"""
Main loop; has no safety constraints added
"""
while True:
    if args.vision:
        tr_tuple = q.get()
        q.task_done()
        trans, rots = tr_tuple[0], tr_tuple[1]

    else:
        ts_node.socket.send_string("sup")
        trans_in, rots_in = ts_node.get_data()
        trans, rots=trans_in[3], rots_in[3]

    ms_pos, ms_rot = env.bot.get_ee_pose()
    
    if not env.bot.initialized:
        env.bot.init_poses(trans, rots, args.vision)
        env.bot.get_cmd_q(trans, rots, args.vision)
    else:
        env.bot.get_cmd_q(trans, rots, args.vision)
thread.join()
    

