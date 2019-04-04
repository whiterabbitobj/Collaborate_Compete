# -*- coding: utf-8 -*-
import os.path



def print_bracketing(info=None, do_upper=True, do_lower=True, center=True):
    """
    Formats a provided statement (INFO) for printing to the cmdline. If provided
    a list, will print each element to a new line, and if provided a string,
    will print a single line.
    """
    mult = 50
    if type(info) is not list and info is not None:
        mult = max(mult, len(info))
        info = [info]
    bracket = "#"
    upper = ("{0}\n{1}{2}{1}".format(bracket*mult, bracket, " "*(mult-2)))
    lower = ("{1}{2}{1}\n{0}".format(bracket*mult, bracket, " "*(mult-2)))
    if do_upper: print(upper)

    if info and center: print('\n'.join([line.center(mult) for line in info]))
    elif info: print('\n'.join(info))

    if do_lower: print(lower)
    return

def check_dir(dir):
    """
    Creates requested directory if it doesn't yet exist.
    """

    if not os.path.isdir(dir):
        os.makedirs(dir)



################################################################################
# The functions below are primarily for use with the Jupyter Notebook attached #
# to this project and can be safely disregarded if not touching the IPYNB file #
################################################################################


from data_handling import Logger
from agent import MAD4PG_Net
import torch
import numpy as np
import matplotlib.pyplot as plt

def print_args(args):
    print('\n'.join(["{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)]))

def print_env_info(state, action, reward):
    print("The agent chooses ACTIONS that look like:\n{}\n".format(action))
    print("The environment returns STATES that look like:\n{}\n".format(state))
    print("The environment returns REWARDS that look like:\n{}".format(reward))
#
def notebook_eval_agent(args, env, filename, num_eps=2):
    eval_agent = MAD4PG_Net(env, args)
    for idx, agent in enumerate(eval_agent.agents):
        weights = torch.load(filename[idx], map_location=lambda storage, loc: storage)
        agent.actor.load_state_dict(weights['actor_dict'])
        agent.critic.load_state_dict(weights['critic_dict'])
        eval_agent.update_networks(agent, force_hard=True)
    args.eval = True
    logger = Logger(eval_agent, args)
    for episode in range(num_eps):
        env.reset()
        state = env.states
        for t in range(200):
            action = eval_agent.act(state, training=False)
            next_state, reward, done = env.step(action)
            state = next_state
            logger.log(reward, eval_agent)
            if np.any(done):
                break
        logger.step(episode)
        eval_agent.new_episode(logger.scores)
    args.eval = False

def test_e(x, box):
    ylow, yhigh, xlow, xhigh = box
    steep_mult = 8

    steepness = steep_mult / (xhigh - xlow)
    offset = (xhigh + xlow) / 2
    midpoint = yhigh - ylow

    x = np.clip(x, 0, xhigh)
    x = steepness * (x - offset)
    e = ylow + midpoint / (1 + np.exp(x))
    return e

def graph_e(box):
    x1 = np.linspace(-2, 2, 150)
    x2 = np.linspace(box[2]-0.5, box[3]+.5, 50)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(x1, test_e(x1, box))
    plt.subplot(212)
    curve = test_e(x2, box)
    plt.yticks(np.linspace(curve.min(), curve.max(), 6))
    plt.plot(x2, curve)
