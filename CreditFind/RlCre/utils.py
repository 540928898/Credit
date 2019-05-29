# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:49:22 2019

@author: Gupeng
"""
import numpy as np

def play_one(env, agent, eps, gamma):
  obs = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done and iters < 2000:
    action = agent.sample_action(obs, eps)
    prev_obs = obs
    obs, reward, done, info = env.step(action)
#    signal.signal(signal.SIGINT, CtrlPannel)
    env.render()

    if done:
      reward = -400

    # update the model
    next = agent.predict(obs)
    assert(len(next.shape) == 1)
    G = reward + gamma*np.max(next)
    agent.update(prev_obs, action, G)

    if reward == 1:
      totalreward += reward
    iters += 1

  return totalreward