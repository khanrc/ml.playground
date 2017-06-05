import tensorflow as tf
import numpy as np
import os
from models import Model
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

batch_size = 100
epoch_n = 20
N = mnist.train.num_examples 
n_iter = N // batch_size
lr = 0.001

# vanilla, NoCD, BN
models = [
    Model(name="vanilla", use_BN=False, use_CD=False, lr=lr),
    Model(name="NoCD", use_BN=True, use_CD=False, lr=lr),
    Model(name="BN", use_BN=True, use_CD=True, lr=lr)
]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# summaries
summary_dir = "summary/"
# if os.path.exists(summary_dir):
#     print('\n[ERROR] Summary directory {} is already exists!\n'.format(summary_dir))
#     exit(1) 

train_writers = [
    tf.summary.FileWriter(summary_dir + 'train/vanilla', sess.graph, flush_secs=5),
    tf.summary.FileWriter(summary_dir + 'train/NoCD', flush_secs=5),
    tf.summary.FileWriter(summary_dir + 'train/BN', flush_secs=5)
]
train_writers = [
    tf.summary.FileWriter(summary_dir + 'test/vanilla', flush_secs=5),
    tf.summary.FileWriter(summary_dir + 'test/NoCD', flush_secs=5),
    tf.summary.FileWriter(summary_dir + 'test/BN', flush_secs=5)
]

for epoch in range(epoch_n):
    for _ in range(n_iter):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        for i in range(3):
            model = models[i]
            writer = train_writers[i]
            print batch_x
            print batch_x.dtype
            print batch_x.shape
            print model.X.shape
            _, cur_summary = sess.run([model.train_op, model.summary_op], feed_dict={model.X: batch_x, model.y: batch_y, model.training: True})
            writer.add_summary(cur_summary, epoch)

    # test run
    for i in range(3):
        model = models[i]
        writer = test_writers[i]
        _, cur_summary = sess.run([model.train_op, model.summary_op], {model.X: mnist.test.images, model.y: mnist.test.labels, model.training: False})
        writer.add_summary(cur_summary, epoch)

    print epoch

