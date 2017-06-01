# coding: utf-8

import tensorflow as tf
import numpy as np
import collections
from affinetransformer import AffineTransformer

"""
좀 예쁘게 짜보자.

40x40 images

구조:
DefaultModel
- AllConvNet(DefaultModel)
- STN(DefaultModel)

defaultmodel 을 사용하는게 좋은지 잘 모르겠다.
그냥 중복코드를 작성할까.

기능:
summary
- histogram of M (STN)
- image summary on affine transform
calc # of params (Default)

"""

def conv(inputs, filters, kernel_size, strides=1, padding='same', activation=tf.nn.relu,
         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name=None):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, padding=padding, activation=activation,
                            kernel_initializer=kernel_initializer, name=name)

def max_pooling(inputs, pool_size=[2,2], strides=2, padding='same', name=None):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding=padding, name=name)

def dense(inputs, units, activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), 
          bias_initializer=None, name=None):
    if bias_initializer:
        return tf.layers.dense(inputs, units, activation=activation, kernel_initializer=kernel_initializer, 
                               bias_initializer=bias_initializer, name=name)
    else:
        return tf.layers.dense(inputs, units, activation=activation, kernel_initializer=kernel_initializer, name=name)

class DefaultModel(object):
    def __init__(self):
        pass

    def _build_nets(self):
        pass

    def _build_graph(self):
        nets = self._build_nets()
        pass


class AllConvNets(object):
    def __init__(self, name="allconvnets", lr=0.001, st_filters=64):
        self.name = name
        self.lr = lr
        self.st_filters = st_filters

        self._build_nets()
        self.num_params = self._count_params()

    def _build_nets(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 1600])
            self.y = tf.placeholder(tf.float32, [None, 10])

            net = tf.reshape(self.X, [-1, 40, 40, 1])

            n_filters = self.st_filters
            for i in range(3):
                net = conv(net, n_filters, kernel_size=[3,3])
                net = conv(net, n_filters, kernel_size=[3,3], strides=2)
                n_filters *= 2

                # filters: 64, 128, 256

            # fc layers
            # [N, 10]
            net = tf.contrib.layers.flatten(net)
            self.logits = dense(net, 10) 
            self.prob = tf.nn.softmax(self.logits)
            
            # mean for mini-batch
            # print '-------------------------'
            # print self.logits.shape
            # print self.y.shape
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1)), tf.float32))
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # summaries
            self.summary_op = tf.summary.merge([
                tf.summary.scalar(self.name+"/loss", self.loss),
                tf.summary.scalar(self.name+"/accuracy", self.accuracy)
            ])

    def _count_params(self):
        #iterating over all variables
        total = 0
        for variable in tf.trainable_variables():
            scope = variable.name.split('/')[0]
            if scope != self.name:
                continue

            local_params = np.prod(variable.shape).value
            total += local_params
            
        return total

# histogram! 
class STNs(object):
    def _make_name(self, name, lr, locnets_type, use_residual_M):
        res_name = "useres" if use_residual_M else "nores"
        return "{}-{}-{}-{}".format(name, lr, locnets_type, res_name)

    def __init__(self, name="stn", lr=0.001, locnets_type='fc', use_residual_M=False):
        self.name = self._make_name(name, lr, locnets_type, use_residual_M)
        self._build_nets(lr=lr, locnets_type=locnets_type, use_residual_M=use_residual_M)
        self.num_params, self.params_counter = self._count_params()

    def _build_nets(self, lr, locnets_type, use_residual_M):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 1600], name='X')
            self.y = tf.placeholder(tf.float32, [None, 10], name='y')

            x_img = tf.reshape(self.X, [-1, 40, 40, 1], name='x_img')

            # localization networks
            with tf.variable_scope("locnets-" + locnets_type):
                loc_net = x_img
                
                if locnets_type == 'conv':
                    # why conv stn does not work?
                    n_filters = 32
                    for i in range(3):
                        loc_net = conv(loc_net, n_filters, kernel_size=[3,3], name="conv{}".format(i))
                        loc_net = max_pooling(loc_net, pool_size=[2,2], strides=2, name="maxpool{}".format(i))
                        n_filters *= 2
                    # 32, 64, 128
                    # 20x20, 10x10, 5x5
                    loc_net = tf.contrib.layers.flatten(loc_net)
                elif locnets_type == 'fc':
                    # cnn => dense
                    loc_net = dense(self.X, 256, activation=tf.nn.relu, name="dense0")
                else:
                    raise 'locnets type Error'

                # print("locnet shape: {}".format(loc_net.shape))
                # 두 방법 모두 2*tanh 같은 activ func 를 사용해줄 필요는 있어 보이는데
                if use_residual_M == False:
                    # residual connection (차이만을 학습) 을 사용하지 않고, 초기값이 identity mapping 이 되도록 한 다음에 학습
                    init = tf.constant_initializer([1., 0., 0., 0., 1., 0.])
                    M = dense(loc_net, 6, kernel_initializer=tf.zeros_initializer, bias_initializer=init, name="dense1")
                else:
                    # residual connection intuition + low learning rate
                    # 근데 이렇게하면 train_loss 는 잘 주는데, test acc 가 이상함... (0.89 정도)
                    # 내생각에는 STN 부분이랑 CNN 부분이랑 밸런스 문제가 있는 것 같은데...
                    # 이건 비주얼라이제이션을 해 봐야 확실히 알 수 있지 않을까 싶음.
                    # => 더 이상한게 위처럼 해도 lr 에 따라서 수렴값이 달라짐... -.-... 그냥 로컬 미니멈의 문제인가...
                    M_diff = dense(loc_net, 6, name="dense1")
                    M = tf.constant([1., 0., 0., 0., 1., 0.]) + M_diff
                
                # grid generator & sampler (affine transform)
                transformer = AffineTransformer(x_img, M)
                net = transformer.transform
            
            with tf.variable_scope("cnn"):
                n_filters = 32
                for i in range(3):
                    net = conv(net, n_filters, kernel_size=[3,3], name="conv{}".format(i))
                    net = max_pooling(net, pool_size=[2,2], strides=2, name="maxpool{}".format(i))
                    n_filters *= 2

                    # filters: 32, 64, 128
                    # size: 20x20, 10x10, 5x5

                # fc layers
                # [N, 10]
                net = tf.contrib.layers.flatten(net)
                self.logits = dense(net, 10) 
                self.prob = tf.nn.softmax(self.logits)
            
            # mean for mini-batch
            with tf.variable_scope("accuracy"):
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1)), tf.float32))
            with tf.variable_scope("loss"):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

            # summaries
            self.summary_op = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                tf.summary.scalar("accuracy", self.accuracy)
            ])

    def _count_params(self):
        #iterating over all variables
        counter = collections.defaultdict(lambda: 0)
        for variable in tf.trainable_variables():
            scope = variable.name.split('/')[0]
            if scope != self.name:
                continue

            sub_scope = variable.name.split('/')[1]
            local_params = np.prod(variable.shape).value
            counter[sub_scope] += local_params
            
        total = sum(counter.values())
        return total, counter






