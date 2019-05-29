# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:40:36 2019

@author: Gupeng
"""
from structure import NeuralNetwork
import numpy as np

class Agent:
  def __init__(self, env, feature_transformer):
    self.env = env
    self.agent = []
    self.feature_transformer = feature_transformer
    for i in range(env.action_space.n):
      nn = NeuralNetwork(feature_transformer.dimensions)
      self.agent.append(nn)

  def predict(self, s):
    X = self.feature_transformer.transform([s])
    return np.array([m.predict(X)[0] for m in self.agent])

  def update(self, s, a, G):
    X = self.feature_transformer.transform([s])
    self.agent[a].train(X, [G])

  def sample_action(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))