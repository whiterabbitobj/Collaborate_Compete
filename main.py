# -*- coding: utf-8 -*-
import numpy as np
from agent import MAD4PG_Net
from environment import Environment
from data_handling import Logger, Saver, gather_args

def main():
    """
    Originall written for Udacity's Collaborate/Compete project:
    https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet

    This project utilizes multi-agent reinforcement learning techniques to
    teach two agents to play table tennis against each other in an environment
    limited to two axis (X, Y).

    The implementation for class uses a variant of OpenAI's MADDPG:
    https://arxiv.org/abs/1706.02275
    but utilizes D4PG as a base algorithm:
    https://arxiv.org/pdf/1804.08617.pdf
    in what I will call MAD4PG.

    The improvements over DDPG include multi-step rollout, and distributional
    reward prediction, for faster and more stable learning.

    A more robust, feature complete and multi-application version of this
    implementation may follow for non-Udacity environments, in a separate
    repository. For instance, more flexible multi-agent handling, instead of the
    hard-coded two agents for this specific application.
    """

    args = gather_args()

    env = Environment(args)

    multi_agent = MAD4PG_Net(env, args)

    saver = Saver(multi_agent, args)

    if args.train:
        train(multi_agent, args, env, saver)
    else:
        eval(multi_agent, args, env)

    return



def train(multi_agent, args, env, saver):
    """
    Train the agent.
    """

    #####################################################
    #                   INIT DATA                       #
    logger = Logger(multi_agent, args, saver.save_dir)
    # Pre-fill the Replay Buffer
    multi_agent.initialize_memory(args.pretrain, env)
    #                                                   #
    #####################################################
    ############################################################################
    #                         Begin training loop                              #
    for episode in range(1, args.num_episodes+1):
        # Begin each episode with a clean environment
        env.reset()
        # Get initial state
        obs = env.states
        # Gather experience until done or max_steps is reached
        while True:
            ##########################################
            #                INTERACT                #
            actions = multi_agent.act(obs)
            next_obs, rewards, dones = env.step(actions)
            multi_agent.store((obs, next_obs, actions, rewards, dones))
            ##########################################
            #                 TRAIN                  #
            multi_agent.learn()
            ##########################################
            #              NEXT TIMESTEP             #
            obs = next_obs
            logger.log(rewards, multi_agent)
            if np.any(dones):
                break
            # Use a variable threshold to end an episode to avoid overly long
            # training after a target score has been reached, beyond which there
            # is little further to learn
            ep_threshold = np.clip(np.random.normal(1.5,.1), 1.3, 2)
            if logger.rewards.max() >= ep_threshold:
                break
        ###################################################
        #              PREP FOR NEXT EPISODE              #
        saver.save(multi_agent)
        logger.step(episode, multi_agent)
        multi_agent.new_episode(logger.scores)
        if len(logger.scores) > 250:
            if np.array(logger.scores[-250:]).mean() > args.quit_threshold:
                break
    #                                                                          #
    ############################################################################
    ##############################################
    #                  CLEANUP                   #
    env.close()
    logger.final(episode, multi_agent)
    saver.save(multi_agent, final=True)
    #                                            #
    ##############################################
    return

def eval(multi_agent, args, env):
    """
    Evaluate the performance of an agent using a saved weights file.
    """

    logger = Logger(multi_agent, args)

    #Begin evaluation loop
    for episode in range(1, args.num_episodes+1):
        # Begin each episode with a clean environment
        env.reset()
        # Get initial state
        obs = env.states
        # Gather experience until done or max_steps is reached
        while True:
            actions = multi_agent.act(obs, training=False)
            next_obs, rewards, dones = env.step(actions)
            obs = next_obs

            logger.log(rewards)
            if np.any(dones):
                break
        # print(logger.scores)
        logger.step(episode, multi_agent)
        multi_agent.new_episode(logger.scores)

    env.close()
    return

if __name__ == "__main__":
    main()
