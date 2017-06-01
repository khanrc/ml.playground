# Spatial Transformer Networks

Jaderberg, Max, Karen Simonyan, and Andrew Zisserman. "Spatial transformer networks." Advances in Neural Information Processing Systems. 2015.

* 이미지가 distorted 되어 있는 경우, 이걸 먼저 transformation 해서 잘 align 해 주고 그 뒤에 classification 을 하겠다는 것.  
* CNN 이 생각보다 이러한 distortion 에 invariant 하지 않다는 단점을 해결하고자 함.
* Main reference:
    * https://kevinzakka.github.io/2017/01/10/stn-part1/
    * https://github.com/kevinzakka/spatial_transformer_network


## ToDo

* [x] tensorboard graph cleaning
* [ ] Summaries
    * [x] loss/accuracy scalar
    * [x] summary_dir arrange
    * [ ] histogram for M
        * 시작값인 identity mapping 에서 어떻게 변하는가
    * [ ] image - input / transformed
    * [ ] text for # of params
    * [ ] compare models & allconv (last)
* allconv/stn 모델 분리하기
    * kkm 님 스타일로 분리해볼까...
    * 그냥 지금처럼 내비두고 그때그때 다른거 불러와서 학습시키는게 낫겠다.
    * 둘다 같이 돌리는건 좀 별로고.
* ETC
    * 그리고 위 영상 (devsummit) 보면 W/b 가 0일때 전혀 학습이 안 되는 경우가 있는데 이찬우님 얘기는 뭐였는지 확인해보고
        * 아마 레이어 하나만 0이면 괜찮은 그런 케이스가 아닐까 싶지만
        * 암튼 백프로파게이션 (직접 해 보면 좋고 아님 말고) 식으로 확인해보자

