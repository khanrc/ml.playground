# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from affinetransformer import AffineTransformer
from utils import *
from models import AllConvNets, STNs
import os 
from argparse import ArgumentParser

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--epochs', default=150, help="Number of training epochs (default: 150)", type=int)
    # parser.add_argument('--batch_size', default=128, help="Batch size (default: 128)", type=int)
    parser.add_argument('--lr', default=0.001, help="Learning rate for ADAM (default: 0.001)", type=float)
    parser.add_argument('--locnet', default='fc', help="Type of localization networks - fc/cnn (default: fc)")
    # parser.add_argument('--residual', default=True, help="Use residual connection for M (default: True)", type=bool)
    # boolean 다루는게 좀 까다로움.
    parser.add_argument('--residual', action='store_true', help="Use residual connection for M", default=False)

    return parser

parser = build_parser()
FLAGS = parser.parse_args()
print("\nParameters:")
for attr, value in sorted(vars(FLAGS).items()):
    print("{}={}".format(attr.upper(), value))
print("")


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
    epoch_n = FLAGS.epochs
    batch_size = 100
    N = X_train.shape[0]
    num_iter = N // batch_size

    # allconv_name = "allconv-{}".format(epoch_n) 
    stn_name = 'stn-zero-cnnonly-{}'.format(epoch_n)

    # build nets
    # tf.reset_default_graph()
    # allconv = AllConvNets()
    # 참 웃긴 놈일세... 0.89~0.97 다양한 값으로 수렴함
    stn = STNs(name=stn_name, locnets_type=FLAGS.locnet, lr=FLAGS.lr, use_residual_M=FLAGS.residual)
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
    # cnn acc/loss 가 큰 의미가 없는 듯. 물론 가끔은 의미있지만...
    # train_cnn_writer = tf.summary.FileWriter(summary_dir + 'train_cnn', flush_secs=10) 
    # test_cnn_writer = tf.summary.FileWriter(summary_dir + 'test_cnn', flush_secs=10)

    for epoch in range(epoch_n):
        np.random.shuffle(train)
        allconv_avg_loss = 0
        stn_avg_loss = 0
        for i in range(0, N, batch_size):
            batch_x = train[i:i+batch_size, :1600]
            batch_y = train[i:i+batch_size, 1600:]
            
            _, stn_cur_loss, stn_summary = sess.run([stn.train_op, stn.loss, stn.summary_op], 
                                                    {stn.X: batch_x, stn.y: batch_y, stn.only_cnn: False})
            stn_avg_loss += stn_cur_loss
            train_writer.add_summary(stn_summary, epoch)

            # stn_cnn_summary = sess.run(stn.summary_op, {stn.X: batch_x, stn.y: batch_y, stn.only_cnn: True})
            # train_cnn_writer.add_summary(stn_cnn_summary, epoch)


        stn_avg_loss /= num_iter
        
        stn_cur_acc, stn_cur_loss, stn_cur_M, stn_summary = sess.run([stn.accuracy, stn.loss, stn.M, stn.summary_op], 
                                                                     {stn.X: X_test, stn.y: y_test, stn.only_cnn: False})
        test_writer.add_summary(stn_summary, epoch)
        
        # stn_cnn_summary = sess.run(stn.summary_op, {stn.X: X_test, stn.y: y_test, stn.only_cnn: True})
        # test_cnn_writer.add_summary(stn_cnn_summary, epoch)

        stn_cur_M = np.mean(stn_cur_M, axis=0)
        
        # print("[{}/{}] (allconv) avg_loss: {:.3f}, acc: {:.3f} | (stn) avg_loss: {:.3f}, acc: {:.3f}". 
        #       format(epoch+1, epoch_n, allconv_avg_loss, allconv_cur_acc, stn_avg_loss, stn_cur_acc))
        print("[{}/{}] (stn) train_avg_loss: {:.3f} | test_loss: {:.3f}, test_acc: {:.3f} | M: {}".
              format(epoch+1, epoch_n, stn_avg_loss, stn_cur_loss, stn_cur_acc, stn_cur_M))

