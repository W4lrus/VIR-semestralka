import airsim
import numpy as np
import time

class AirSimEnv():
    def __init__(self , takeoff = True):
        self.client = airsim.MultirotorClient()   # connect to airsim simulator
        self.client.confirmConnection()
        self.client.enableApiControl(True)   # enable control over python
        self.client.armDisarm(True)
        self.client.simPause(False)

        if takeoff : self.client.takeoffAsync().join()


    def get_obs(self):
        kinematics = self.client.simGetGroundTruthKinematics()  # acc, vel, pos and ori data
        pos = kinematics.position
        ori = kinematics.orientation
        col = self.client.simGetCollisionInfo().has_collided

        response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)

        self.state = {'pos': pos , 'ori': ori , 'img' : img_rgb , 'col' : col}
        return self.state


    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)  # enable control over python
        self.client.armDisarm(True)
        return self.get_obs()


    def step(self , vel_vector ,drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom , duration = 1):  #sets drone velocity for duration in seconds
        '''drivetrain = airsim.DrivetrainType.ForwardOnly makes the drone move only forward. If drone is asked to move to side, he will first rotate, then move forward.
        drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom just moves drone in a specified direction without turning'''
        (x, y, z) = vel_vector

        self.client.simPause(False)
        self.client.moveByVelocityAsync(x,y,z, duration , drivetrain).join()
        time.sleep(duration)
        self.client.simPause(True)
        return self.get_obs()

    def set_velocity(self , vel_vector , drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom , duration = 1):

        (x, y, z) = vel_vector

        self.client.moveByVelocityAsync(x,y,z, duration, drivetrain).join()
        return self.get_obs()
