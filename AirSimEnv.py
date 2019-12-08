import airsim
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

    def get_pos(self):
        kinematics = self.client.simGetGroundTruthKinematics()  # acc, vel, pos and ori data
        pos = kinematics.position
        ori = kinematics.orientation
        return pos , ori

    def get_rgb_img(self): # return rgb image as numpy array
        response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        return img_rgb

    def get_obs(self):
        col = self.client.simGetCollisionInfo().has_collided
        img_rgb = self.get_rgb_img()
        pos, ori = self.get_pos()
        self.state = {'pos': pos, 'ori': ori, 'img': img_rgb, 'col': col} # update state
        return self.state

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)  # enable control over python
        self.client.armDisarm(True)
        return self.get_obs()

    def step(self , vel_vector , duration = 1):  # sets drone velocity for duration in seconds
        '''drivetrain = airsim.DrivetrainType.ForwardOnly makes the drone move only forward.
        If drone is asked to move to side, he will first rotate, then move forward.
        drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom just moves drone in a specified direction without turning'''

        (vx, vy, vz) = vel_vector
        self.client.simPause(False) # unfreeze
        self.client.moveByVelocityAsync(vx, vy, vz, duration, self.drivetrain).join()  #join makes the function wait for end of task
        self.client.simPause(True) # freeze
        return self.get_obs()

    def step_z(self, vel_vector, duration=1):  # sets velocity using x,y vector and sets z constant
        (vx, vy, z) = vel_vector
        self.client.simPause(False)
        self.client.moveByVelocityZAsync(vx, vy, z, duration, self.drivetrain).join()
        self.client.simPause(True)
        return self.get_obs()

    def set_velocity(self, vel_vector, duration=1, wait=True):
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

    def hover(self):
        self.client.hoverAsync().join()
