import numpy as np
import torch as T
import my_utils
import NNQvalues
import os
from AirSimEnv import AirSimEnv

goal = 300

def train(env, policy, params):
    print('Training started')

    init_coords = (0, 0, -10)

    policy_optim = T.optim.Adam(policy.parameters(), lr=params["policy_lr"], weight_decay=params["weight_decay"],
                                eps=1e-4)

    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_terminals = []
    batch_ctr = 0
    batch_rew = 0

    for i in range(params["iters"]):
        print('Episode', i, 'started')
        env.reset()
        env.move_to(init_coords)

        step_ctr = 0

        done = False
        old_action = np.array([0, 0])
        while not done:
            img = env.get_rgb_img()

            action = policy.sample_action(img.cuda()).detach().cpu()  # generate action using img
            batch_states.append(img.cpu())  # add image to state history
            batch_actions.append(action)  # add action to action history

            action = old_action + action[0].numpy()
            old_action = action

            velx = action[0].item()
            vely = action[1].item()
            new_state = env.step_z((velx, vely, -10), duration=params["step_length"])  # use generated action to move

            reward, done = compute_reward(new_state, velx, vely, step_ctr)  # compute reward in new state
            batch_rew += reward
            step_ctr += 1

            if step_ctr == params['maxsteps']:  # new epoch if too many steps (memory is not infinite)
                done = True

            batch_rewards.append(my_utils.to_tensor(np.asarray(reward, dtype=np.float32), True))  # add reward to reward history
            batch_terminals.append(done)  # add terminal state position to history

        batch_ctr += 1
        print(env.get_pos())

        if batch_ctr == params["batchsize"]:
            batch_states = T.cat(batch_states)
            batch_actions = T.cat(batch_actions)
            batch_rewards = T.cat(batch_rewards)

            batch_rewards = (batch_rewards - batch_rewards.mean()) / batch_rewards.std()  # scale rewards
            batch_advantages = calc_advantages_MC(params["gamma"], batch_rewards, batch_terminals)  # get advantages

            update_ppo(policy, policy_optim, batch_states.to(params["device"]), batch_actions.to(params["device"]),
                       batch_advantages.to(params["device"]), params["ppo_update_iters"])

            print("Episode {}/{}, loss_V: {}, loss_policy: {}, mean ep_rew: {}".
                  format(i, params["iters"], None, None, batch_rew / params["batchsize"]))

            # Finally reset all batch lists
            batch_ctr = 0
            batch_rew = 0

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_terminals = []

            if params["device"] == 'cuda':
                T.cuda.empty_cache()  # free memory

            if i % 100 == 0 and i > 0:
                sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "agents/{}_{}_{}_pg.p".format(env.__class__.__name__, policy.__class__.__name__, params["ID"]))
                T.save(policy, sdir)
                print("Saved checkpoint at {} with params {}".format(sdir, params))

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

    print('stds', policy.log_std.detach().cpu().numpy()[0])


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


def compute_reward(state, vx, vy, t):
    v = 2
    position = state['pos']
    collision = state['col']
    reward = 0
    done = False

    reward += position[1] - v*t - (vy**2)/10 + 0.5*vx

    if position[1] > goal:
        reward += 1000
        done = True

    if collision:
        reward -= 500
        done = True

    return reward, done


if __name__ == "__main__":
    params = {"iters": 200, "batchsize": 2, "maxsteps": 80, "step_length": 0.5, "device": 'cuda', "gamma": 0.995, "policy_lr": 0.005,
              "weight_decay": 0.0001, "ppo_update_iters": 6, "train": True}
    print('Connecting to AirSim Environment')
    env = AirSimEnv(freeze=True, takeoff=False)
    print('AirSim environment initiated')
    policy = NNQvalues.Policy(tanh=True, std_fixed=False).to(params["device"])
    print('Policy created')
    train(env, policy, params)

    env.hover()
