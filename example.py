from AirSimEnv import AirSimEnv
import matplotlib.pyplot as plt
import time

env = AirSimEnv(freeze=False)  # connect to airsim and freeze the simulation

state = env.get_obs()  # get state info
position = state['pos']  # pull coordinates of drone from state variable
orientation = state['ori']  # pull orientation of drone from state variable

vector = (0, 0, -1)  # velocity tuple (vx , vy , vz)  ===  +Z IS DOWN ===
duration = 5  # how long to hold the velocity

new_state = env.step(vector, duration)
collision = new_state['col']  # True or False for collision

env.hover() # stay in place

image = new_state['img']  # pull rgb image in numpy array format from the state variable
plt.imshow(image)
plt.show()

time.sleep(2)

new_state = env.set_velocity_z((0, 1, -5), duration=5)  # set x and y velocity and keep z=-5 constant
env.hover()

env.reset()  # reset drone to start position (0,0,0)







