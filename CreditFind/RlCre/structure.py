# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:49:22 2019

@author: Gupeng
"""
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, D):
        eta = 0.1
        self.W = tf.Variable(tf.random_normal(shape=(D, 1)), name='w')
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        # make prediction and cost
        Y_hat = tf.reshape(tf.matmul(self.X, self.W), [-1])
        err = self.Y - Y_hat
        cost = tf.reduce_sum(tf.pow(err,2))

        # ops we want to call later
        self.train_op = tf.train.GradientDescentOptimizer(eta).minimize(cost)
        self.predict_op = Y_hat

        # start the session and initialize params
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def train(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X: X})
