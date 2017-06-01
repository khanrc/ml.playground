# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from affinetransformer import AffineTransformer
from utils import *
from models import AllConvNets, STNs
import os 


if __name__ == "__main__":
    # 40x40
    # labels are dense (no one-hot)
    mnist_cluttered = np.load('../cluttered_mnist_data/mnist_sequence1_sample_5distortions5x5.npz')

    X_train = mnist_cluttered['X_train'] # 10000
    y_train = mnist_cluttered['y_train']
    X_valid = mnist_cluttered['X_valid'] # 1000
    y_valid = mnist_cluttered['y_valid']
    X_test = mnist_cluttered['X_test'] # 1000
    y_test = mnist_cluttered['y_test']

    y_train = one_hot(y_train)
    y_valid = one_hot(y_valid)
    y_test = one_hot(y_test)

    train = np.concatenate([X_train, y_train], axis=1)
    # print train.shape
    assert train.shape == (10000, 1610)

    # hyperparams
    epoch_n = 10
    batch_size = 100
    N = X_train.shape[0]
    num_iter = N // batch_size

    # allconv_name = "allconv-{}".format(epoch_n)
    stn_name = 'stn-{}'.format(epoch_n)

    # build nets
    # tf.reset_default_graph()
    # allconv = AllConvNets()
    # 참 웃긴 놈일세... 0.89~0.97 다양한 값으로 수렴함
    stn = STNs(name=stn_name, locnets_type='fc', lr=0.0005, use_residual_M=True)
    summary_dir = "summary/{}/".format(stn.name)
    if os.path.exists(summary_dir):
        print('\n[ERROR] Summary directory {} is already exists!\n'.format(summary_dir))
        exit(1) 


    print('-------------------------------------------')
    # print('[AllConvNets] # of params: {}'.format(allconv.num_params)) 
    print('[   STNs    ] {}'.format(stn.name))
    print('[   STNs    ] # of params: {}'.format(stn.num_params))
    print('[   STNs    ] counter: {}'.format(stn.params_counter))
    print('-------------------------------------------')

    # train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(summary_dir + 'train', sess.graph, flush_secs=10)
    test_writer = tf.summary.FileWriter(summary_dir + 'test', flush_secs=10)

    for epoch in range(epoch_n):
        np.random.shuffle(train)
        allconv_avg_loss = 0
        stn_avg_loss = 0
        for i in range(0, N, batch_size):
            batch_x = train[i:i+batch_size, :1600]
            batch_y = train[i:i+batch_size, 1600:]
            
            # _, allconv_cur_loss, allconv_summary = sess.run([allconv.train_op, allconv.loss, allconv.summary_op], {allconv.X: batch_x, allconv.y: batch_y})
            _, stn_cur_loss, stn_summary = sess.run([stn.train_op, stn.loss, stn.summary_op], {stn.X: batch_x, stn.y: batch_y})
            # allconv_avg_loss += allconv_cur_loss
            stn_avg_loss += stn_cur_loss
            # train_writer.add_summary(allconv_summary, epoch)
            train_writer.add_summary(stn_summary, epoch)
        
        # allconv_avg_loss /= num_iter
        stn_avg_loss /= num_iter
        
        # allconv_cur_acc, allconv_summary = sess.run([allconv.accuracy, allconv.summary_op], {allconv.X: X_test, allconv.y: y_test})
        stn_cur_acc, stn_cur_loss, stn_summary = sess.run([stn.accuracy, stn.loss, stn.summary_op], {stn.X: X_test, stn.y: y_test})
        # test_writer.add_summary(allconv_summary, epoch)
        test_writer.add_summary(stn_summary, epoch)
        
        # print("[{}/{}] (allconv) avg_loss: {:.3f}, acc: {:.3f} | (stn) avg_loss: {:.3f}, acc: {:.3f}".
        #       format(epoch+1, epoch_n, allconv_avg_loss, allconv_cur_acc, stn_avg_loss, stn_cur_acc))

        print("[{}/{}] (stn) train_avg_loss: {:.3f} | test_loss: {:.3f}, test_acc: {:.3f}".format(epoch+1, epoch_n, stn_avg_loss, stn_cur_loss, stn_cur_acc))

