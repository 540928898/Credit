# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:39:16 2019

@author: Gupeng
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler
class FeatureTransformer:
  def __init__(self, env):
    #obs_examples = np.array([env.observation_space.sample() for x in range(20000)])

    obs_examples = np.random.random((20000, 4))
    print(obs_examples.shape)
    scaler = StandardScaler()
    scaler.fit(obs_examples)

    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = FeatureUnion([
            ("cart_position", RBFSampler(gamma=0.02, n_components=500)),
            ("cart_velocity", RBFSampler(gamma=1.0, n_components=500)),
            ("pole_angle", RBFSampler(gamma=0.5, n_components=500)),
            ("pole_velocity", RBFSampler(gamma=0.1, n_components=500))
            ])
    feature_examples = featurizer.fit_transform(scaler.transform(obs_examples))
    print(feature_examples.shape)

    self.dimensions = feature_examples.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer

  def transform(self, observations):
    scaled = self.scaler.transform(observations)
    return self.featurizer.transform(scaled)
