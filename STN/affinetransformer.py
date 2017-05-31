##### coding: utf-8

import tensorflow as tf

"""
Spatial Transformer Networks 는 크게 3개로 구성된다.
1. localization networks
2. (affine) grid generator
3. (bilinear) sampler

괄호친건 다른형태로도 가능하지만 여기서는 이렇게 구현할 것이라는 의미다.

localization networks 에서는 affine transform 에 필요한 transform matrix M 을 생성할 것이고,
여기서는 이 M 을 받아서 affine grid generator 를 생성하고 이를 사용해서 bilinear sampling 을 수행한다.
"""

def get_pixel_value(img, x, y):
    """
    이 함수의 목적은,
    ret[k, i, j] = img[k, y[i, j], x[i, j]]
    를 수행하는것이다.
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    
    batch_idx = tf.range(0, batch_size) # [B]
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1)) # [B, 1, 1]
    b = tf.tile(batch_idx, (1, height, width)) # [B, height, width]

    indices = tf.stack([b, y, x], 3) # (B, height, width, 3)
    
    # 결과적으로, indices 는 indices[k, i, j] = (batch_idx, y, x) 를 갖게 된다.
    # 그러면 img[k, i, j, c] 로부터, tf.gather_nd(img, indices) 를 사용해서
    # 원하는 좌표들을 뽑아낼 수 있다.
    
    # gather_nd 에 대해 좀더 자세히 설명하자면, (http://devdocs.io/tensorflow~python/tf/gather_nd)
    # tf.gather_nd(params, indices) 로 이루어져 있고,
    # params[k, i, j] 라 했을 때
    # ret[k, i, j] = params[indices[k, i, j]] 가 된다.
    # 여기서 한걸음 더 나아가서 인덱싱 뿐만 아니라 슬라이싱도 가능한데,
    # indices[k, i, j] 의 값이 3개가 아닌 경우 (즉 rank 가 3이 아닌 경우), 자연스럽게 params 를 slicing 해서 가져오게 된다.
    # 결과적으로 ret 의 shape 은 indices 의 shape 을 기본으로 하고, 거기에 element 가 indices[k, i, j] 에서 가져온게 들어오게 되므로
    # indices.shape 보다 더 커질 수 있다.
    # 최소로 작아지는 경우, indices.shape[:-1] 의 크기가 된다 - indices.shape[-1] 은 엘리먼트 하나를 가리킬 수 있으므로.
    # 말로 설명하면 더 어려운 것 같고, api docs 의 예제들을 보면 이해가 될 것이다.

    return tf.gather_nd(img, indices)

# 네이밍이라던가 구조가 좀 구리긴 한데...
# H, W 는 target (H, W) 를 의미하고,
# self.height, self.width 는 source (H, W) 를 의미한다.
# 리팩토링이 좀 필요할 듯
class AffineTransformer:
    def __init__(self, X, M, out_size=None):
        # 일반적으로 이 X 는 minibatch images 가 됨
        self.X = X
        # 여기서 알아둘 것! X.shape, X.get_shape(), tf.shape(X) 의 차이!
        # tf.shape(X) 는 ops 다. 즉, 리턴이 텐서로 나온다. 
        # 반면 X.shape 이나 X.get_shape 은 뭐랄까.. imperative 함수다. 리턴값이 텐서가 아니다.
        # 따라서 아래와 같이 그래프 내에서 shape 을 활용하고 싶다면 tf.shape 을 사용해야 한다!
#         self.batch_size, self.height, self.width, self.channels = tf.shape(self.X) #self.X.shape, tf.shape(self.X)
#         self.batch_size, self.height, self.width, self.channels = self.X.shape
        self.batch_size = tf.cast(tf.shape(self.X)[0], tf.int32)
        self.height = int(self.X.shape[1])
        self.width = int(self.X.shape[2])
        self.channels = int(self.X.shape[3])
        # batch_size 만 유동적임. 나머지까지 tf.shape 로 하면 오히려 문제가 생김 (이 네트웤은 고정 인풋에 대한 네트웤이므로)
        # 구체적으로는 dense layer 에서 문제가 생김. input dense layer 의 input layer 의 unit 개수가 ? 가 되어서
        # 컴파일이 안 된다.
        # 만약 FCN 같이 유동 인풋을 커버하는 네트웤이라면 tf.shape 로 해야 할 듯
#         self.height = tf.cast(tf.shape(self.X)[1], tf.int32)
#         self.width = tf.cast(tf.shape(self.X)[2], tf.int32)
#         self.channels = tf.cast(tf.shape(self.X)[3], tf.int32)
        if out_size is not None:
            H, W = out_size
        else:
            H, W = self.height, self.width
        
        sampling_grid = self.affine_grid_generator(H, W, M)
        batch_grids = self.affine_transform(sampling_grid, H, W, M)
        self.transform = self.bilinear_sampler(batch_grids)
        

    def affine_grid_generator(self, H, W, M):
        """
        H: height of batch_grid
        W: width of batch_grid
        M: affine transform matrix

        H/W 를 조정함으로써 up/downsampling 이 가능함.
        """

        # MAKE SAMPLING GRID
        # 먼저 M 을 batch_size 만큼 뻥튀기해야함
        # => 생각해보니 여기서는 애초에 localization network 에서 batch size 만큼의 그 각각의 데이터포인트에 맞는 M 을 생성해줄거임!

        # create normalized 2D grid
        # 사실 여기서 flatten 을 꼭 해줄 필요는 없는듯. => 해줘야 한다
        # matmul 에서 어차피 flatten 을 해줘야하기 때문에.
        # matmul 이 곱하는 두 텐서가 같은 rank 임을 요구함.
        # matmul 에 대해서 공부할 필요가 있는듯... 어케 곱해지는건지 도통 모르겠음. 3차원 행렬곱.
        x = tf.linspace(-1.0, 1.0, W)
        y = tf.linspace(-1.0, 1.0, H)
        x_t, y_t = tf.meshgrid(x, y)
        x_t_flat = tf.reshape(x_t, [-1]) # flatten
        y_t_flat = tf.reshape(y_t, [-1]) # flatten

        # 이제 이걸 논문에 나온것처럼 (xt, yt, 1) 형태로 재구성 
#         ones = tf.ones_like(x_t_flat)
        ones = tf.ones(W*H) # 이건 왜 안 될까?
        sampling_grid = tf.stack([x_t_flat, y_t_flat, ones], axis=0)
        assert sampling_grid.shape == (3, W*H)

        # 이 샘플링 그리드가 배치 사이즈만큼 필요함!
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)
        sampling_grid = tf.tile(sampling_grid, tf.stack([self.batch_size, 1, 1]))
        assert sampling_grid.shape[1:] == (3, W*H)

        return sampling_grid


    def affine_transform(self, sampling_grid, H, W, M):
        # AFFINE TRANSFORM - 이거 밖으로 빼는게 맞을거 같기도... 흠... 여기에있는게 맞나...
        # transform sampling grid
        M = tf.reshape(M, [-1, 2, 3])
        M = tf.cast(M, tf.float32) # matmul required float32
        batch_grids = tf.matmul(M, sampling_grid) # [batch_size, 2, 3] * [batch_size, 3, W*H] = [batch_size, 2, W*H]
        assert batch_grids.shape[1:] == (2, W*H) # 여기서 2 는 x,y 임!

        # reshape to (batch_size, 2, H, W)
        # np 로 짤때는 (batch_size, H, W, 2) 로 reshape 했는데, 굳이 그럴 필요가 없는거 같다.
        batch_grids = tf.reshape(batch_grids, [self.batch_size, 2, H, W])

        # batch_grids[k, :, i, j] = (x, y)
        # 이 (x, y) 는 (i, j) 를 어디에서 가져와야되는지를 의미
        # 즉, target[i, j] = source[y, x]
        # 따라서 여기서 i, j 는 [0, H], [0, W] 의 범위에 있고, y, x 는 [-1, 1] 에 존재.

        return batch_grids


    def bilinear_sampler(self, batch_grids):
        _, _, H, W = batch_grids.shape

        x_s = batch_grids[:, 0, :, :]
        y_s = batch_grids[:, 1, :, :]
#         print x_s.shape
        assert x_s.shape[1:] == (H, W) and y_s.shape[1:] == (H, W)

        x = ((x_s + 1.)/2.) * tf.cast(self.width, tf.float32)
        y = ((y_s + 1.)/2.) * tf.cast(self.height, tf.float32)

        # x0, y0, x1, y1 type 을 int 로 바꿔줘야함?
        # ㅇㅇ 바까줘야함. stack 할때 다 타입이 같아야 하네. => 하나의 텐서로 만드는거니까 당연
        # 그리고 직관적으로 int 가 맞다.
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, self.width-1)
        x1 = tf.clip_by_value(x1, 0, self.width-1)
        y0 = tf.clip_by_value(y0, 0, self.height-1)
        y1 = tf.clip_by_value(y1, 0, self.height-1)

        Ia = tf.cast(get_pixel_value(self.X, x0, y0), tf.float32)
        Ib = tf.cast(get_pixel_value(self.X, x0, y1), tf.float32)
        Ic = tf.cast(get_pixel_value(self.X, x1, y0), tf.float32)
        Id = tf.cast(get_pixel_value(self.X, x1, y1), tf.float32)

        # calc delta for interpolation weight coef
        # 여기선 다시 float32 가 필요함...
        # recast as float for delta calculation
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        # add dims for addition
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        # compute output 
        # out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        out = wa*Ia + wb*Ib + wc*Ic + wd*Id

        return out