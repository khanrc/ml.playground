# coding: utf-8

import tensorflow as tf
import numpy as np
from vgg19 import VGG
import imageutils as im


# hyperparams
content_layers = ['conv4_2'] # 보통 하나만 쓰지만 그냥 scalability 를 위해...
# style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'] # 별 차이 없는듯

def gram_matrix(features):
    c = features.shape[-1].value
    flat = tf.reshape(features, [-1, c]) # N=1 이라서 이렇게 함
    return tf.matmul(tf.transpose(flat), flat)

class StyleTransfer(object):
    def __init__(self, content, style, config):
        self.vgg = VGG()

        # preprocessing
        content = self.vgg.preproc(content)
        style = self.vgg.preproc(style)
        content = np.expand_dims(content, axis=0)
        style = np.expand_dims(style, axis=0)

        self.content = content
        self.style = style
        self.cfg = config
        self.build_net()

    def build_net(self):
        # get features
        # style_image = tf.placeholder(tf.float32, [None] + list(self.style.shape[1:]))
        # content_image = tf.placeholder(tf.float32, [None] + list(self.content.shape[1:]))
        content_image = tf.constant(self.content, tf.float32)
        style_image = tf.constant(self.style, tf.float32)
        synthesis_image = tf.Variable(initial_value=self.content, trainable=True, dtype=tf.float32) # init by content image

        style_features = self.vgg.build_net(style_image)
        content_features = self.vgg.build_net(content_image)
        synthesis_features = self.vgg.build_net(synthesis_image)

        # calculate loss
        # style loss
        style_loss = 0.
        for i, sl in enumerate(style_layers):
            style_f = style_features[sl]
            syn_f = synthesis_features[sl]
            h, w, c = style_f.shape[1:]
            
            G = gram_matrix(style_f)
            A = gram_matrix(syn_f)
            N = c.value # number of feature map
            M = h.value * w.value # feature map size

            cur_loss = tf.reduce_sum(tf.square(G-A))
            cur_loss /= (4 * N**2 * M**2)
            style_loss += 0.2 * cur_loss # 0.2: style weight per layers

        # content loss
        content_loss = 0.
        for i, cl in enumerate(content_layers):
            content_f = content_features[cl] # P
            syn_f = synthesis_features[cl] # F

            h, w, c = content_f.shape[1:]
            N = c.value # number of feature map
            M = h.value * w.value # feature map size
            
            if self.cfg.content_loss_type == 1: # original paper
                K = 0.5
            elif self.cfg.content_loss_type == 2: # Artistic style transfer for videos
                K = 1. / (N * M)
            elif self.cfg.content_loss_type == 3: # maybe Preserving Color in Neural Artistic Style Transfer ?
                K = 1. / (2. * N**0.5 * M**0.5)

            cur_loss = K * tf.reduce_sum(tf.square(content_f - syn_f))
            content_loss += cur_loss

        # tv loss
        h, w, c = synthesis_image.shape[1:]
        tv_y_size = ((h-1)*w*c).value
        tv_x_size = (h*(w-1)*c).value
        tv_loss_y = tf.nn.l2_loss(synthesis_image[:, 1:, :, :] - synthesis_image[:, :-1, :, :]) / tv_y_size
        tv_loss_x = tf.nn.l2_loss(synthesis_image[:, :, 1:, :] - synthesis_image[:, :, :-1, :]) / tv_x_size
        tv_loss = 2 * (tv_loss_y + tv_loss_x)

        # total loss
        self.total_loss = self.cfg.content_weight*content_loss + self.cfg.style_weight*style_loss + self.cfg.tv_weight*tv_loss
        self.content_loss = content_loss
        self.style_loss = style_loss
        self.tv_loss = tv_loss
        self.content_image = content_image
        self.style_image = style_image
        self.synthesis_image = synthesis_image

        
