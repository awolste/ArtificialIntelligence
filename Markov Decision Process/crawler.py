# crawler.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this file, you should test your Q-learning implementation on the robot crawler environment 
that we saw in class. It is suggested to test your code in the grid world environments before this one.

The package `matplotlib` is needed for the program to run.


The Crawler environment has discrete state and action spaces
and provides both of model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
  

Once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
"""


# use random library if needed
import random

import numpy as np


def q_learning(env, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    gamma = 0.95
    DECREASE_EVERY = 500

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    # maximum number of training iterations
    max_iterations = 5000
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    old_q = np.zeros([NUM_STATES, NUM_ACTIONS])
    q = np.zeros([NUM_STATES, NUM_ACTIONS])
    #########################

### Please finish the code below ##############################################
###############################################################################
    count = 0
    episodes = 0
    a = 0
    rewards = 0
    while True:
        episodes += 1
        s = env.reset()
        delta = 0
        while True:
            # check if too many iterations
            if count > max_iterations:
                logger.log(count, v, pi)
                return pi
            # decrease eps every 10% of iterations
            if count % DECREASE_EVERY == 0:
                eps -= .1
            p = np.random.random()
            if p < eps:
                a = random.randint(0, NUM_ACTIONS - 1)
            else:
                max_q_val = q[s][0]
                for action in range(1, NUM_ACTIONS):
                    if q[s][action] > max_q_val:
                        a = action
            s_, r, terminal, info = env.step(a)
            if terminal:
                target = r
            else:
                action_values = [0] * NUM_ACTIONS
                # loop through next actions to find highest q val
                for a_ in range(NUM_ACTIONS):
                    action_values[a_] = q[s_][a_]
                target = r + gamma * max(action_values)
            # update current state action q value
            rewards += r
            q[s][a] = (1 - alpha) * (q[s][a]) + alpha * target
            delta = max(delta, abs(q[s][a] - old_q[s][a]))
            s = s_
            if terminal:
                break
            max_q_val = q[s][0]
            pi[s] = 0
            for action in range(1, NUM_ACTIONS):
                # print(s, a)
                if q[s][action] > max_q_val:
                    pi[s] = action  # pi is action of the highest q value of all actions from that state
                    max_q_val = q[s][action]
                v[s] = max_q_val
            count += 1
        if delta < .001:
            break
        print(delta)
        old_q = q.copy()
        logger.log(count, v, pi)
###############################################################################
    return None


if __name__ == "__main__":
    from app.crawler import App
    import tkinter as tk
    
    algs = {
        "Q Learning": q_learning,
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()
