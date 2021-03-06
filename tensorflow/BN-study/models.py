# coding: utf-8

import tensorflow as tf


class Model(object):
    def __init__(self, name, use_BN, use_CD, lr):
        # use_BN
        # use_CD: control dependencies
        self.name = name
        self._build_nets(name, use_BN, use_CD, lr)


    def _build_nets(self, name, use_BN, use_CD, lr=0.001):
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
            self.y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
            self.training = tf.placeholder(tf.bool, name='training')

            net = tf.reshape(self.X, [-1, 28, 28, 1])

            n_filters = 32
            for i in range(3):
                with tf.variable_scope("conv{}".format(i)):
                    # conv
                    net = tf.layers.conv2d(net, n_filters, [3,3], padding='same', use_bias=not use_BN)
                    if use_BN:
                        net = tf.layers.batch_normalization(net, training=self.training)
                    net = tf.nn.relu(net)

                with tf.variable_scope("maxpool{}".format(i)):
                    # max_pool
                    net = tf.layers.max_pooling2d(net, [2,2], strides=2, padding='same')

                n_filters *= 2
                # [14, 14, 32], [7, 7, 64], [4, 4, 128]
                # 2048

            with tf.variable_scope("dense"):
                net = tf.contrib.layers.flatten(net)
                self.logits = tf.layers.dense(net, 10)
            
            with tf.variable_scope("prob"):
                self.prob = tf.nn.softmax(self.logits)
            
            with tf.variable_scope("accuracy"):
                self.accuracy = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1))
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

            with tf.variable_scope("loss"):
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
                self.loss = tf.reduce_mean(self.loss)

            with tf.variable_scope("train_op"):
                if use_CD:
                    # same as summaries, get_collection must specify scope!
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name+'/')
                    with tf.control_dependencies(update_ops):
                        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
                else:
                    self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

            # add moving averages to histogram
            histograms = []
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name+'/'):
                if "moving_" in var.name:
                    histograms.append(tf.summary.histogram(var.name, var))
                    tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
            # now, we can get moving averages by tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, scope=name+'/') 

            # summaries
            # Caution: When design multiple models in a single graph,
            # `tf.summary.merge_all` function tries merging every summaries of models.
            self.summary_op = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                tf.summary.scalar("accuracy", self.accuracy)
            ] + histograms)