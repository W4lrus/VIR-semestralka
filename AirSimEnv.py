import airsim
import torch as T
import numpy as np

'''drivetrain = airsim.DrivetrainType.ForwardOnly makes the drone move only forward.
If drone is asked to move to side, he will first rotate, then move forward.
drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom just moves drone in a specified direction without turning'''


class AirSimEnv:
    def __init__(self, takeoff=True, dt=0, freeze=False):  # if freeze=True sim is frozen if possible
        self.client = airsim.MultirotorClient()   # connect to airsim simulator
        self.client.confirmConnection()
        self.client.enableApiControl(True)   # enable control over python
        self.client.armDisarm(True)

        self.stepping = freeze
        self.takeoff = takeoff

        self.state = self.get_obs()  # init state
        if dt:
            self.drivetrain = airsim.DrivetrainType.ForwardOnly
        else:
            self.drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom
        if takeoff:
            self.unfreeze()
            self.client.takeoffAsync().join()
        self.client.simPause(self.stepping)

    def get_pos(self, numpy=True):
        self.unfreeze()
        pos = self.client.simGetGroundTruthKinematics().position
        self.client.simPause(self.stepping)
        if numpy:
            pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        return pos

    def get_ori(self, numpy=True):
        self.unfreeze()
        ori = self.client.simGetGroundTruthKinematics().orientation
        self.client.simPause(self.stepping)
        if numpy:
            ori = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])
        return ori

    def get_rgb_img(self, tensor=True):  # return rgb image as numpy array or torch tensor
        self.unfreeze()
        response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        self.client.simPause(self.stepping)

        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        if tensor:
            img_rgb = T.from_numpy(img_rgb).float()
            img_rgb = img_rgb.transpose(0, 2).transpose(1, 2)
            s = img_rgb.shape
            img_rgb = img_rgb.view(-1, s[0], s[1], s[2])
        return img_rgb

    def get_obs(self):  # return everything as numpy arrays
        self.unfreeze()
        col = self.client.simGetCollisionInfo().has_collided
        self.client.simPause(self.stepping)

        img_rgb = self.get_rgb_img(tensor=False)
        ori = self.get_ori()
        pos = self.get_pos()
        self.state = {'pos': pos, 'ori': ori, 'img': img_rgb, 'col': col}  # update state
        return self.state

    def reset(self):
        self.unfreeze()
        self.client.reset()
        self.client.enableApiControl(True)  # enable control over python
        self.client.armDisarm(True)

        self.state = self.get_obs()  # init state
        if self.takeoff:
            self.client.takeoffAsync().join()
        self.client.simPause(self.stepping)
        return self.get_obs()

    def step(self, vel_vector, duration=1):  # sets drone velocity for duration in seconds
        self.unfreeze()
        self.set_velocity_z(vel_vector, duration)
        self.freeze()
        return self.get_obs()

    def step_z(self, vel_vector, duration=1):  # sets velocity using x,y vector and sets z constant
        self.unfreeze()
        self.set_velocity_z(vel_vector, duration)
        self.freeze()
        return self.get_obs()

    def set_velocity(self, vel_vector, duration=1, wait=True):
        self.unfreeze()
        (vx, vy, vz) = vel_vector
        if wait:
            self.client.moveByVelocityAsync(vx, vy, vz, duration, self.drivetrain).join()
        else:
            self.client.moveByVelocityAsync(vx, vy, vz, duration, self.drivetrain)

    def set_velocity_z(self, vel_vector, duration=1, wait=True):  # sets velocity using x,y vector and sets z constant
        (vx, vy, z) = vel_vector
        if wait:
            self.client.moveByVelocityZAsync(vx, vy, z, duration, self.drivetrain).join()
        else:
            self.client.moveByVelocityZAsync(vx, vy, z, duration, self.drivetrain)

    def move_to(self , coordinates, v=5):
        self.unfreeze()
        (x, y, z) = coordinates
        self.client.moveToPositionAsync(x, y, z, v).join()
        self.client.simPause(self.stepping)

    def hover(self):
        self.unfreeze()
        self.client.hoverAsync().join()
        self.client.simPause(self.stepping)

    def freeze(self):
        self.client.simPause(True)  # unfreeze

    def unfreeze(self):
        self.client.simPause(False)  # unfreeze