import airsim
import numpy as np
import torch as T
import numpy as np


class AirSimEnv():
    def __init__(self, takeoff=True, dt=0, freeze=False):
        self.client = airsim.MultirotorClient()   # connect to airsim simulator
        self.client.confirmConnection()
        self.client.enableApiControl(True)   # enable control over python
        self.client.armDisarm(True)
        self.client.simPause(freeze)
        self.state = self.get_obs() # init state
        if dt:
            self.drivetrain = airsim.DrivetrainType.ForwardOnly
        else:
            self.drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom
        if takeoff:
            self.client.takeoffAsync().join()

    def get_pos(self, numpy=True):
        pos = self.client.simGetGroundTruthKinematics().position
        if numpy:
            pos = np.array([pos.x_val , pos.y_val , pos.z_val])
        return pos

    def get_ori(self, numpy=True):
        ori = self.client.simGetGroundTruthKinematics().orientation
        if numpy:
            ori = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])
        return ori

    def get_rgb_img(self, tensor=True):  # return rgb image as numpy array or torch tensor
        response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        if tensor:
            img_rgb = T.from_numpy(img_rgb).float()
            img_rgb.transpose(2, 0, 1)
        return img_rgb

    def get_obs(self):  # return everything as numpy arrays
        col = self.client.simGetCollisionInfo().has_collided
        img_rgb = self.get_rgb_img(tensor=False)
        ori = self.get_ori()
        pos = self.get_pos()
        self.state = {'pos': pos, 'ori': ori, 'img': img_rgb, 'col': col}  # update state
        return self.state

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)  # enable control over python
        self.client.armDisarm(True)

        self.client.simPause(freeze)
        self.state = self.get_obs()  # init state
        if takeoff:
            self.client.takeoffAsync().join()
        return self.get_obs()

    def step(self , vel_vector , duration = 1):  # sets drone velocity for duration in seconds
        '''drivetrain = airsim.DrivetrainType.ForwardOnly makes the drone move only forward.
        If drone is asked to move to side, he will first rotate, then move forward.
        drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom just moves drone in a specified direction without turning'''

        (vx, vy, vz) = vel_vector
        self.unfreeze
        self.client.moveByVelocityAsync(vx, vy, vz, duration, self.drivetrain).join()  #join makes the function wait for end of task
        self.freeze
        return self.get_obs()

    def step_z(self, vel_vector, duration=1):  # sets velocity using x,y vector and sets z constant
        (vx, vy, z) = vel_vector
        self.unfreeze
        self.client.moveByVelocityZAsync(vx, vy, z, duration, self.drivetrain).join()
        self.freeze
        return self.get_obs()

    def set_velocity(self, vel_vector, duration=1, wait=True):
        self.unfreeze
        (vx, vy, vz) = vel_vector
        if wait:
            self.client.moveByVelocityAsync(vx, vy, vz, duration, self.drivetrain).join()
        else:
            self.client.moveByVelocityAsync(vx, vy, vz, duration, self.drivetrain)

    def set_velocity_z(self, vel_vector, duration=1, wait=True):  # sets velocity using x,y vector and sets z constant
        self.unfreeze
        (vx, vy, z) = vel_vector
        if wait:
            self.client.moveByVelocityZAsync(vx, vy, z, duration, self.drivetrain).join()
        else:
            self.client.moveByVelocityZAsync(vx, vy, z, duration, self.drivetrain)

    def move_to(self , coordinates, v=5):
        (x, y, z) = coordinates
        self.client.moveToPositionAsync(x, y, z, v).join()

    def hover(self):
        self.client.hoverAsync().join()

    def freeze(self):
        self.client.simPause(False)  # unfreeze

    def unfreeze(self):
        self.client.simPause(True)  # unfreeze

