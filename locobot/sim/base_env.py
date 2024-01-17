import numpy as np
import pybullet as p
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from airobot.utils.pb_util import create_pybullet_client
from yacs.config import CfgNode as CN

import locobot
from locobot.sim.locobot import Locobot


class BaseEnv:
    def __init__(self, gui=True, realtime=False, opengl_render=False, n_substesps=10):
        self.n_substeps = n_substesps
        self.pb_client = create_pybullet_client(gui=gui, realtime=realtime, opengl_render=opengl_render)
        self.pb_client.setAdditionalSearchPath(locobot.LIB_PATH.joinpath('assets').as_posix())

    def reset(self):
        self.pb_client.resetSimulation()
        self.pb_client.setGravity(0, 0, -9.8)
        self.pb_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        self.plane_id = self.pb_client.loadURDF('plane/plane.urdf', [0, 0, -0.001])
        self.bot_id = self.pb_client.loadURDF('locobot_description/locobot.urdf', [0, 0, 0], useFixedBase=1)
        self.bot = Locobot(self.bot_id)
        self.bot.set_locobot_camera_pan_tilt(0., 0.6)
        fp_cam_pos, fp_cam_ori = self.bot.get_locobot_camera_pose()
        # first-person camera
        self.fp_cam = self.create_camera(pos=fp_cam_pos, ori=fp_cam_ori)
        # third-person camera
        self.tp_cam = self.create_camera(pos=np.array([0, 0, 4.5]), ori=np.array([0.707, -0.707, -0., -0.]))
        self.add_boundary()
        self.pb_client.configureDebugVisualizer(self.pb_client.COV_ENABLE_RENDERING, 1)
        return None

    def add_boundary(self):
        half_thickness = 0.02
        half_height = 0.2
        half_length = 3
        self.pb_client.load_geom(shape_type='box',
                                 size=[half_length, half_thickness, half_height],
                                 mass=0,
                                 rgba=[0.6, 0.4, 0.2, 1],
                                 base_pos=[0, half_length, half_height])
        self.pb_client.load_geom(shape_type='box',
                                 size=[half_thickness, half_length, half_height],
                                 mass=0,
                                 rgba=[0.6, 0.4, 0.2, 1],
                                 base_pos=[half_length, 0, half_height])
        self.pb_client.load_geom(shape_type='box',
                                 size=[half_length, half_thickness, half_height],
                                 mass=0,
                                 rgba=[0.6, 0.4, 0.2, 1],
                                 base_pos=[0, -half_length, half_height])
        self.pb_client.load_geom(shape_type='box',
                                 size=[half_thickness, half_length, half_height],
                                 mass=0,
                                 rgba=[0.6, 0.4, 0.2, 1],
                                 base_pos=[-half_length, 0, half_height])

    def get_fp_images(self):
        fp_cam_pos, fp_cam_ori = self.bot.get_locobot_camera_pose()
        self.fp_cam.set_cam_ext(pos=fp_cam_pos, ori=fp_cam_ori)
        return self.fp_cam.get_images()

    def get_tp_images(self):
        return self.tp_cam.get_images()

    def create_camera(self, pos, ori, cfg=None):
        if cfg is None:
            cfg = self._get_default_camera_cfg()
        cam = RGBDCameraPybullet(cfgs=cfg, pb_client=self.pb_client)
        cam.set_cam_ext(pos=pos, ori=ori)
        return cam

    def forward_simulation(self, nsteps=None):
        if nsteps is None:
            nsteps = self.n_substeps
        for i in range(nsteps):
            p.stepSimulation()

    def _get_default_camera_cfg(self):
        _C = CN()
        _C.ZNEAR = 0.01
        _C.ZFAR = 10
        _C.WIDTH = 640
        _C.HEIGHT = 480
        _C.FOV = 60
        _ROOT_C = CN()
        _ROOT_C.CAM = CN()
        _ROOT_C.CAM.SIM = _C
        return _ROOT_C.clone()
