import numpy as np
import torch as T
import my_utils
import airsim
import os
import NNQvalues
import tempfile

goal = [20, 20, 20]
startdistance_goal = np.sqrt(goal[0]^2 + goal[1]^2 + goal[2]^2)


def train(client, policy, params):
    initX = 0
    initY = 0
    initZ = -10

    policy_optim = T.optim.Adam(policy.parameters(), lr=params["policy_lr"], weight_decay=params["weight_decay"],
                                eps=1e-4)

    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_new_states = []
    batch_terminals = []
    batch_ctr = 0
    batch_rew = 0

    step_ctr = 0

    for i in range (params["iters"]):
        # Start flying
        client.takeoffAsync().join()
        client.moveToPositionAsync(initX, initY, initZ, 5).join()

        # # Debug get images
        # for i in range(2):
        #     responses = client.simGetImages([
        #         airsim.ImageRequest("0", airsim.ImageType.DepthVis)])  # depth visualization image
        #         # airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True),  # depth in perspective projection
        #         # airsim.ImageRequest("1", airsim.ImageType.Scene),  # scene vision image in png format
        #         # airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  # scene vision image in uncompressed RGBA array
        #     tmp_dir = "C:\\Users\Filip\\PycharmProjects\\LearningDrone\\airsim_drone"
        #     for idx, response in enumerate(responses):
        #
        #         filename = os.path.join(tmp_dir, str(idx))
        #
        #         if response.pixels_as_float:
        #             print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        #             airsim.write_pfm(os.path.normpath(filename + str(i) + '.pfm'), airsim.get_pfm_array(response))
        #         elif response.compress:  # png format
        #             print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        #             airsim.write_file(os.path.normpath(filename + str(i) + '.png'), response.image_data_uint8)
        #
        #     client.moveByVelocityAsync(0, 5, 0, 2).join()

        done = False
        while not done:
            # Get image for sample_action
            responses = client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
            for idx, response in enumerate(responses):
                # img1d = np.array(responses[0].image_data_float, dtype=np.float)
                # img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3) # 144x256x3

            img_rgb.transpose(2, 0, 1)
            img_rgb = T.from_numpy(img_rgb).float()
            # Sample action from policy
            action = policy.sample_action(img_rgb.flatten()).detach()
            # Record transition
            batch_states.append(client.getMultirotorState().kinematics_estimated.position)
            batch_actions.append(action.numpy())
            action = action.numpy()

            # Step action, rotate and move
            if action[0] < 0:
                action[0] = 0
            t = action[0].item()
            client.rotateByYawRateAsync(40, t) # Fixed rotation speed, action[0] = rotation span
            velx = action[1].item()
            vely = action[2].item()
            client.moveByVelocityAsync(velx, vely, 0, 0.5).join() # Moving in the xy plane, fixed z

            reward, done = compute_reward(client)

            batch_rew += reward

            step_ctr += 1

            # Record transition
            batch_rewards.append(my_utils.to_tensor(np.asarray(reward, dtype=np.float32), True))
            batch_new_states.append(client.getMultirotorState().kinematics_estimated.position)
            batch_terminals.append(done)

        # Completed episode
        batch_ctr += 1

        if batch_ctr == params["batchsize"]:

            batch_states = T.cat(batch_states)
            batch_actions = T.cat(batch_actions)
            batch_rewards = T.cat(batch_rewards)

            # Scale rewards
            batch_rewards = (batch_rewards - batch_rewards.mean()) / batch_rewards.std()

            # Calculate episode advantages
            batch_advantages = calc_advantages_MC(params["gamma"], batch_rewards, batch_terminals)

            update_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, params["ppo_update_iters"])

            print("Episode {}/{}, loss_V: {}, loss_policy: {}, mean ep_rew: {}".
                  format(i, params["iters"], None, None, batch_rew / params["batchsize"]))

            # Finally reset all batch lists
            batch_ctr = 0
            batch_rew = 0

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_new_states = []
            batch_terminals = []

        # Saved learned model
        # if i % 100 == 0 and i > 0:
        #     sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                         "agents/{}_{}_{}_pg.p".format(env.__class__.__name__, policy.__class__.__name__, params["ID"]))
        #     T.save(policy, sdir)
        #     print("Saved checkpoint at {} with params {}".format(sdir, params))

        # Reset
        client.reset()
        # If computer vision mode enabled
        # # Reset position
        # # currently reset() doesn't work in CV mode. Below is the workaround
        # client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)


def update_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, update_iters):
    log_probs_old = policy.log_probs(batch_states, batch_actions).detach()
    c_eps = 0.2

    # Do ppo_update
    for k in range(update_iters):
        log_probs_new = policy.log_probs(batch_states, batch_actions)
        r = T.exp(log_probs_new - log_probs_old)
        loss = -T.mean(T.min(r * batch_advantages, r.clamp(1 - c_eps, 1 + c_eps) * batch_advantages))
        policy_optim.zero_grad()
        loss.backward()
        policy.soft_clip_grads(3.)
        policy_optim.step()


def calc_advantages_MC(gamma, batch_rewards, batch_terminals):
    N = len(batch_rewards)

    # Monte carlo estimate of targets
    targets = []
    for i in range(N):
        cumrew = T.tensor(0.)
        for j in range(i, N):
            cumrew += (gamma ** (j - i)) * batch_rewards[j]
            if batch_terminals[j]:
                break
        targets.append(cumrew.view(1, 1))
    targets = T.cat(targets)

    return targets


def transform_input(responses):
    """Transforms output of simGetImages to one 84x84 image"""
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = 255/np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

    from PIL import Image
    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((84, 84)).convert('L'))

    return im_final


def compute_reward(client):
    # distance from goal
    quad_state = client.getMultirotorState().kinematics_estimated.position
    collision_info = client.simGetCollisionInfo()
    reward = 0
    done = False
    travel_dist = np.sqrt(np.power(goal[0], 2) + np.power(goal[1], 2))        # distance from initial position (0,0) to the goal in plane

    # distance_goal = np.sqrt((goal[0] - quad_state.x_val)*(goal[0] - quad_state.x_val) + (goal[1] - quad_state.y_val)*(goal[1] - quad_state.y_val))
    distance_goal = np.sqrt(np.power(goal[0] - quad_state.x_val, 2) + np.power(goal[1] - quad_state.y_val, 2))
    
    if distance_goal < 1:
        reward += 1000
        done = True
    
    reward += travel_dist - distance_goal
    # reward -= 1
    if collision_info.has_collided:
        reward -= 1000

    return reward, done

if __name__=="__main__":
    # params = {"iters": 500000, "batchsize": 1, "gamma": 0.995, "policy_lr": 0.0007, "weight_decay": 0.0001, "ppo": True,
    #           "ppo_update_iters": 6, "animate": False, "train": True}

    params = {"iters": 100, "batchsize": 1, "gamma": 0.995, "policy_lr": 0.0007, "weight_decay": 0.0001, "ppo_update_iters": 6, "train": True}

    # Airsim environment
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    policy = NNQvalues.Policy()
    train(client, policy, params)

    # Quit
    # that's enough fun for now. let's quit cleanly
    client.enableApiControl(False)
