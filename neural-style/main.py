# coding: utf-8

import tensorflow as tf
import numpy as np
from vgg19 import VGG
import cv2
import matplotlib.pyplot as plt
import imageutils as im
import os
from argparse import ArgumentParser
from models import StyleTransfer
import time

# content_weight = 1e-3 # alpha
# style_weight = 1 # beta
# tv_weight = 0 # total variation denoising

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content', help='content image path', required=True)
    parser.add_argument('--style', help='style image path', required=True)
    parser.add_argument('--output', help='result image path (default: res/`content`_`style`_`max_size`_`content_weight`.jpg)')
    parser.add_argument('--max_size', default=512, help='max size of image (default: 512)', type=int)
    parser.add_argument('--content_weight', default=1e-3, help='content weight (default: 1e-3)', type=float)
    parser.add_argument('--style_weight', default=1., help='style weight (default: 1)', type=float)
    parser.add_argument('--tv_weight', default=0, help='total variation denoising weight (default: 0)', type=float)
    parser.add_argument('--content_loss_type', default=3, help='content loss type (default: 3)', type=int)
    parser.add_argument('--n_iter', default=1000, help='the number of iteration (default: 1000)', type=int)
    # parser.add_argument('--gpu', default=None, help='use gpu number (default: CUDA_VISIBLE_DEVICES)', type=int)

    return parser

def output_path(content, style, dir_name='res'):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    get_base = lambda x: os.path.splitext(os.path.basename(x))[0]
    content = get_base(content)
    style = get_base(style)
    fn = "{}_{}.jpg".format(content, style)
    return os.path.join(dir_name, fn)


if __name__ == "__main__":
    parser = build_parser()
    FLAGS = parser.parse_args()

    if FLAGS.output is None:
        FLAGS.output = output_path(FLAGS.content, FLAGS.style)

    print("\nParameters:")
    for attr, value in sorted(vars(FLAGS).items()):
        print("{}={}".format(attr, value))
    print("")

    # image load
    content = im.load(FLAGS.content, max_size=FLAGS.max_size)
    style = im.load(FLAGS.style, max_size=FLAGS.max_size)

    # build nets & preproc
    # with tf.device('/gpu:1'): # works?
    st = StyleTransfer(content=content, style=style, config=FLAGS)

    # optimize
    with tf.Session() as sess:
        st_time = time.time()
        sess.run(tf.global_variables_initializer())
        it = 0

        def callback(tl, cl, sl, tvl, image):
            global it
            it += 1
            if it == 1 or it % 100 == 0:
                print('[{}/{}] {:.2e} = {:.2e} + {:.2e} + {:.2e} (total_loss = content_loss + style_loss + tv_loss)'.
                      format(it, FLAGS.n_iter, tl, cl*FLAGS.content_weight, sl*FLAGS.style_weight, tvl*FLAGS.tv_weight))
                # im_show(vgg.unproc(image[0]))

        # 텐서플로에서는 L-BFGS-B 를 제공하지 않음
        # scipy 에서 제공하는걸 갖다씀
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(st.total_loss, method='L-BFGS-B', options={'maxiter': FLAGS.n_iter})

        optimizer.minimize(sess, {st.style_image: st.style, st.content_image: st.content}, 
                           fetches=[st.total_loss, st.content_loss, st.style_loss, st.tv_loss, st.synthesis_image], loss_callback=callback)

        elapsed = time.time() - st_time
        print("Optimization time: {:.1f}s".format(elapsed))

        final_image = sess.run(st.synthesis_image)[0]
        final_image = st.vgg.unproc(final_image)
        final_image = np.clip(final_image, 0.0, 255.0)
        
        im.save(final_image, FLAGS.output)


