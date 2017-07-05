import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import collections

mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
slim = tf.contrib.slim

def one_hot(dense, n_class=10):
    N = dense.shape[0]
    ret = np.zeros([N, n_class])
    ret[np.arange(N), dense] = 1
    return ret

class AffineGenerator():
    def __init__(self, mnist):
        from keras.preprocessing.image import ImageDataGenerator
        
        self.mnist = mnist
        self.datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
        self.train_x = np.reshape(self.mnist.train.images, [-1, 28, 28, 1])
        self.train_y = self.mnist.train.labels

    def generate(self, batch_size=64):
        cnt = 0
        batch_n = self.train_x.shape[0] // batch_size
        for x, y in self.datagen.flow(self.train_x, self.train_y, batch_size=batch_size):
            ret_x = x.reshape(-1, 784)
            yield ret_x, y

            cnt += 1
            if cnt == batch_n:
                break

class BestModel(object):
    def __init__(self, lr):
        X = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])
        training = tf.placeholder(tf.bool)

        net = tf.reshape(X, [-1, 28, 28, 1])
        n_filters = 32
        bn_param = {'is_training': training, 'scale': True, 'decay': 0.99}
        with slim.arg_scope([slim.conv2d], kernel_size=[3,3],
                            normalizer_fn=slim.batch_norm, normalizer_params=bn_param):
            for _ in range(3):
                net = slim.conv2d(net, n_filters)
                net = slim.conv2d(net, n_filters)
                net = slim.max_pool2d(net, kernel_size=[2,2], padding='same')
                net = slim.dropout(net, keep_prob=0.7, is_training=training)
                n_filters *= 2

        flat = slim.flatten(net)
        logits = slim.fully_connected(flat, 10, activation_fn=None)
        prob = tf.nn.softmax(logits)

        # add predict ops for majority voting ensemble
        predict = tf.argmax(logits, axis=1)
        correct = tf.equal(predict, tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(loss)

        # must do this even with slim
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        
        # for interaction
        self.X = X
        self.y = y
        self.training = training
        self.predict = predict
        self.accuracy = accuracy
        self.loss = loss
        self.train_op = train_op

N = mnist.train.num_examples
dq = collections.deque()
datagen=AffineGenerator(mnist)

configs = [
    {'lr': 0.0004, 'batch_size':  50, 'epoch_n': 100},
    {'lr': 0.0007, 'batch_size': 100, 'epoch_n': 140},
    {'lr':  0.001, 'batch_size': 100, 'epoch_n': 100},
    {'lr':  0.002, 'batch_size': 200, 'epoch_n': 120},
    {'lr':  0.003, 'batch_size': 300, 'epoch_n': 150}
]
epoch_n = 100

pred_vote = np.zeros([mnist.test.num_examples, 10])
for i, cfg in enumerate(configs):
    tf.reset_default_graph()
    model = BestModel(cfg['lr'])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Train model{} ... {}".format(i, cfg))
        for epoch in range(cfg['epoch_n']):
            avg_loss = 0.
            avg_acc = 0.

            n_iter = 0
            for batch_x, batch_y in datagen.generate(batch_size=cfg['batch_size']):
                feed_dict = {model.X: batch_x, model.y: batch_y, model.training: True}
                _, cur_acc, cur_loss = sess.run([model.train_op, model.accuracy, model.loss], feed_dict=feed_dict)
                avg_acc += cur_acc
                avg_loss += cur_loss
                n_iter += 1

            avg_acc /= n_iter
            avg_loss /= n_iter

            feed_dict = {model.X: mnist.test.images, model.y: mnist.test.labels, model.training: False}
            test_acc, test_loss = sess.run([model.accuracy, model.loss], feed_dict=feed_dict)
            
            print("[{:2}/{}] (train) acc: {:.2%}, loss: {:.3f} | (test) acc: {:.2%}, loss: {:.3f}".
                  format(epoch+1, cfg['epoch_n'], avg_acc, avg_loss, test_acc, test_loss))
    
        # use last model for ensemble
        feed_dict = {model.X: mnist.test.images, model.y: mnist.test.labels, model.training: False}
        test_acc, test_pred = sess.run([model.accuracy, model.predict], feed_dict=feed_dict)
        dq.append(test_acc)
        pred_vote += one_hot(test_pred)
        print

print("last test accs for each model")
for i, acc in enumerate(dq):
    print("model{}: {:.2%}".format(i, acc))

ensemble_acc = np.average(np.argmax(mnist.test.labels, axis=1) == np.argmax(pred_vote, axis=1))
print("ensemble: {:.2%}".format(ensemble_acc))