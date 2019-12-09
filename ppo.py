import numpy as np
import torch as T
import my_utils
import airsim
import os
import NNQvalues
import tempfile

from AirSimEnv import AirSimEnv

goal = np.array([20, 20, -20])
start_goal_distance = np.linalg.norm(goal)

def train(env, policy, params):

    init_coords = (0, 0, -10)

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

    for i in range(params["iters"]):
        env.reset()
        env.move_to(init_coords)

        done = False
        while not done:
            img = env.get_rgb_img()

            action = policy.sample_action(img.flatten()).detach() # TODO make policy CNN

            batch_states.append(img)
            batch_actions.append(action)
            action = action.numpy()

            if action[0] < 0:  # TODO make descrete actions, or remake policy to support continuous actions
                action[0] = 0
            t = action[0].item()
            #client.rotateByYawRateAsync(40, t)  # Fixed rotation speed, action[0] = rotation span # TODO implement rotation interface if needed
            velx = action[1].item()
            vely = action[2].item()
            vels = (velx,vely,0)
            env.set_velocity(vels, 0.5)

            reward, done = compute_reward(env.get_obs()) # TODO mby better reward function??
            batch_rew += reward
            step_ctr += 1

            batch_rewards.append(my_utils.to_tensor(np.asarray(reward, dtype=np.float32), True))
            batch_new_states.append(env.get_rgb_img())
            batch_terminals.append(done)

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
        env.reset()

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
    img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

    from PIL import Image
    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((84, 84)).convert('L'))

    return im_final


def compute_reward(state):
    position = state['pos']
    collision = state['col']
    reward = 0
    done = False
    travel_dist = np.linalg.norm(goal[:2]) # x and y distance from origin to goal

    distance_goal = np.linalg.norm(goal[:2] - position[:2])

    if distance_goal < 1:
        reward += 1000
        done = True

    reward += travel_dist - distance_goal
    # reward -= 1
    if collision:
        reward -= 1000
        done = True

    return reward, done


if __name__ == "__main__":
    params = {"iters": 100, "batchsize": 1, "gamma": 0.995, "policy_lr": 0.0007, "weight_decay": 0.0001,
              "ppo_update_iters": 6, "train": True}
    env = AirSimEnv()

    policy = NNQvalues.Policy()

    train(env, policy, params)

    env.hover()