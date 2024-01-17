import numpy as np
import zmq
import queue
import glob
#import h5py
from scipy.spatial.transform import Rotation as R

def recv_array(socket, flags=0, copy=True, track=False):
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

def recv_trak_star1(socket, flags=0):
    msg = socket.recv(flags=flags)
    data = np.fromstring(msg, sep=" ").reshape(4,6)
    trans = data[:,:3]

    rots = R.from_euler("xyz", data[:,3:])
    return trans, rots

def recv_trak_star2(socket, flags=0):
    msg = socket.recv(flags=flags)
    data = np.fromstring(msg, sep=" ").reshape(8,6)
    trans = data[:,:3]

    rots = R.from_euler("xyz", data[:,3:])
    return trans, rots

context = zmq.Context()

class ZmqNode:
    def __init__(self, name, url, decoder):
        self.name = name
        self.url = url
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(self.url)
        self.decoder = decoder

    def get_data(self):
        data = self.decoder(self.socket)
        return data

ts_node1 = ZmqNode("action", "tcp://127.0.0.1:5551", recv_trak_star1)
ts_node2 = ZmqNode("action", "tcp://127.0.0.1:5551", recv_trak_star2)
"""
def store_data(filename, packet_q, run):
    f = h5py.File(filename, "w")
    while run or not packet_q.empty():
        try:
            packet = packet_q.get(timeout=1/30.)
            packet_q.task_done()

            for pk in packet.keys():
                data = packet[pk]
                if pk not in f.keys():
                    f.create_dataset(pk, data=data[None,...], maxshape=(None,) + data.shape)
                else:
                    f[pk].resize(f[pk].shape[0] + 1, axis=0)
                    f[pk][-1] = data

        except queue.Empty:
            pass
    f.close()
"""