# BatchNorms

* Why SNN claims that BN does not work on FNN?
* In our toy experiments, why BN does not work on very early stage of training?

## Control dependencies

tf.control\_dependencies 의 효과를 알아본다.

* tensorboard 의 graph 에서 batch\_norm 의 moving variables 에 train\_op 와의 control dependency edge 가 빠져 있는 것으 확인할 수 있음
* tensorboard 의 histogram 에서 control dependency 를 연결해주지 않을 경우 moving variables 들이 update 되지 않는 것을 확인할 수 있음

### Debug BNs

* 이를 위해 @wookayin 님의 발표자료를 많이 참조함
    * https://wook.kr/cv.html#talks 참조
* Tips
    * scope: `name + /` 로 쓰자. scope 가 그냥 prefix 찾는 것 같다. // 를 안써주면 다른것들까지 딸려올 수 있음
    * Why moving_average_variables() empty?
        * https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/tf.moving_average_variables.md
        * `ExponentialMovingAverage` object 가 생성되고 `apply()` 메소드로 variables 에 적용되었을 때 이 variables 들이 `GraphKeys.MOVING_AVERAGE_VARIABLES` 에 등록됨
        * `tf.layers.batch_normalization()` 에서는 위 플로우를 따르지 않아서 여기에 등록되지 않음
    * 만약 특정한 variable 을 가져오고 싶다면, `tf.contrib.framework.get_variables()` 를 활용할 수 있음

## Self-Normalizing Networks

SNN 에 대해서 살펴본다.

* BN 을 쓸 수 없을 때 쓸만한 녀석인 것 같다.
* 자세히 살펴보진 않았음.

## Why BN works wrong on very early stage of training?

```
ReLU
[1/10] (train) acc: 67.64%, loss: 1.112 | (test) acc: 89.53%, loss: 0.364
[2/10] (train) acc: 91.32%, loss: 0.292 | (test) acc: 93.40%, loss: 0.203
[3/10] (train) acc: 95.22%, loss: 0.167 | (test) acc: 96.03%, loss: 0.136
[4/10] (train) acc: 95.74%, loss: 0.152 | (test) acc: 96.43%, loss: 0.112
[5/10] (train) acc: 96.74%, loss: 0.102 | (test) acc: 97.36%, loss: 0.080
[6/10] (train) acc: 96.78%, loss: 0.102 | (test) acc: 95.96%, loss: 0.122
[7/10] (train) acc: 97.08%, loss: 0.097 | (test) acc: 97.92%, loss: 0.068
[8/10] (train) acc: 97.54%, loss: 0.080 | (test) acc: 97.70%, loss: 0.064
[9/10] (train) acc: 97.80%, loss: 0.076 | (test) acc: 97.97%, loss: 0.066
[10/10] (train) acc: 97.42%, loss: 0.075 | (test) acc: 98.29%, loss: 0.054

ReLU_BN
[1/10] (train) acc: 83.16%, loss: 0.582 | (test) acc: 11.35%, loss: 2.346
[2/10] (train) acc: 95.96%, loss: 0.143 | (test) acc: 11.35%, loss: 3.016
[3/10] (train) acc: 97.38%, loss: 0.097 | (test) acc: 11.35%, loss: 3.291
[4/10] (train) acc: 97.26%, loss: 0.089 | (test) acc: 11.35%, loss: 3.144
[5/10] (train) acc: 97.70%, loss: 0.075 | (test) acc: 12.03%, loss: 2.633
[6/10] (train) acc: 98.04%, loss: 0.066 | (test) acc: 23.43%, loss: 2.025
[7/10] (train) acc: 97.92%, loss: 0.074 | (test) acc: 54.03%, loss: 1.151
[8/10] (train) acc: 97.92%, loss: 0.065 | (test) acc: 64.23%, loss: 1.041
[9/10] (train) acc: 98.34%, loss: 0.057 | (test) acc: 89.34%, loss: 0.319
[10/10] (train) acc: 98.08%, loss: 0.057 | (test) acc: 97.73%, loss: 0.082
```

각 epoch 당 iteration = 50, batch\_size=100.

* iteration 이 300 쯤 되고 나서야 제대로 작동하기 시작하며 500쯤 되야 제 성능이 나옴
* BN 을 쓰지 않으면 훨씬 빨리 acc 가 올라감
* train loss 는 오히려 BN 을 써야 더 빨리 떨어짐
* 가설 1. gamma/beta 가 저정도 되야 (iter=500) 자리를 잡음
    * 어차피 같이 학습되는건데 딱히 얘만 더 의존적일 이유가 있나...
    * 체크는 해 보자