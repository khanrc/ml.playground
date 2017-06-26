import tensorflow as tf
import numpy as np

CONTENT_PATH = 'data/flash.jpg'
CONTENT_LAYER = 'relu4_2'
STYLE_PATH = 'data/style1.jpg'
STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

# hyperparams
content_weight = 5
style_weight = 100
tv_weight = 500 # total variational denoising
learning_rate = 5. # 10???

def imread(path):
    return scipy.misc.imread(path)
def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)

vgg = VGG()

content = imread(CONTENT_PATH)