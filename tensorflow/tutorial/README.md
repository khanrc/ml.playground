# TensorFlow + Keras tutorial

based on python 2.7, TF 1.2.1, Keras 2.0.6.

> pascal05

## Contents

* `mnist-tutorial.ipynb`
    * mnist 99.7% - dropout, batchnorm, data augmentation, ensemble, wrong case checking
    * tf.slim
* `mnist-tutorial-tensorboard.ipynb`
    * TensorBoard tutorial
    * (tf.summary.) scalar, histogram, image, text
    * variable scoping
    * folder naming - 다른 폴더에 같은 scope 로 저장하면 겹쳐서 볼 수 있다.
* `inputpipe.ipynb`
    * TensorFlow input pipeline with flower dataset
    * (tf.train.) batch, batch_join, shuffle_batch, shuffle_batch_join 등 테스트
    * 언제 뭘 써야 할지
    * 결론은 shuffle_batch_join + num_epochs 이 짱인 듯
        * 다만 이 경우 에퐄이 늘어났을 때 체크를 할 수 있는지 확인해 봐야 함
        * local_variable 이 생기는 것 같으니 아마 체크가 가능할 듯
* `Keras-BNmodel.ipynb`
    * flower dataset + file list input
    * Very easy in Keras!
    * sequential api, functional api
* `Keras-transferlearning.ipynb`
    * Transfer Learning with Keras
    * 근데 85%정도까지밖에 안 나옴... TF 예제에서는 90%이상 가는 것 같은데. 나중에 더 해보자.

### WIP

* `inputpipe-flower.ipynb`
    * 귀찮아서 하다 맘
    * inputpipe.ipynb 코드를 옮겨 담기만 하면 됨