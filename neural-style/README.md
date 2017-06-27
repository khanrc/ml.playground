# Neural style

based on python 2.7, tensorflow 1.2

## Checklist

* Why origianl loss function does not work?
    * 다른건 다 오리지널 페이퍼 그대로 구현했는데 content_loss 가 이 경우 밸런스가 안 맞음.
    * content_weight 를 조절해서 맞춰줄 수야 있겠지만... 그냥 원본 페이퍼대로 안 되는 이유가 좀 궁금함.
* Fully TF implementation
    * 이렇게 하면 오히려 느려지는거 아닌가?
    * vgg 에 보내서 style/content feature 뽑고 하는 삽질을 매 옵티마이즈 때마다 하는 것 같은데
* placeholder to variable
    * style, content 매번 플레이스홀더로 넣어줄 필요가 없을 것 같은데.
    * 이렇게 바꾸면 속도도 빨라지나? (매번 다시 계산하지 않나?)
    * => 이건 똑같을 것 같기는 한데, 실험은 해봐야 함
* Why works differently?
    * refer code 와 동일한 것 같은데 결과가 살짝 다름.
    * 이유가 뭘까?

### Gatcha

* VGG in the computational graph
    * 이거 학습할 때 vgg network 도 학습되는거 아닌가...?
    * => 이건 아님! VGG network 에 파라메터들을 tf.variable 로 넣어준게 아니기때문에 trainable 하지 않다.

## Results

<h3>flash.jpg</h3><p><img src='images/content/flash.jpg' height='192'> <img src='res/flash_starry-night.jpg' height='192'> <img src='images/style/starry-night.jpg' height='192'> </p><p><img src='images/content/flash.jpg' height='192'> <img src='res/flash_kandinsky2.jpg' height='192'> <img src='images/style/kandinsky2.jpg' height='192'> </p><p><img src='images/content/flash.jpg' height='192'> <img src='res/flash_kandinsky.jpg' height='192'> <img src='images/style/kandinsky.jpg' height='192'> </p><p><img src='images/content/flash.jpg' height='192'> <img src='res/flash_picasso.jpg' height='192'> <img src='images/style/picasso.jpg' height='192'> </p><p><img src='images/content/flash.jpg' height='192'> <img src='res/flash_the_scream.jpg' height='192'> <img src='images/style/the_scream.jpg' height='192'> </p><p><img src='images/content/flash.jpg' height='192'> <img src='res/flash_monet.jpg' height='192'> <img src='images/style/monet.jpg' height='192'> </p><p><img src='images/content/flash.jpg' height='192'> <img src='res/flash_seated-nude.jpg' height='192'> <img src='images/style/seated-nude.jpg' height='192'> </p><p><img src='images/content/flash.jpg' height='192'> <img src='res/flash_shipwreck.jpg' height='192'> <img src='images/style/shipwreck.jpg' height='192'> </p><p><img src='images/content/flash.jpg' height='192'> <img src='res/flash_woman-with-hat-matisse.jpg' height='192'> <img src='images/style/woman-with-hat-matisse.jpg' height='192'> </p><h3>gbk.jpg</h3><p><img src='images/content/gbk.jpg' height='192'> <img src='res/gbk_starry-night.jpg' height='192'> <img src='images/style/starry-night.jpg' height='192'> </p><p><img src='images/content/gbk.jpg' height='192'> <img src='res/gbk_kandinsky2.jpg' height='192'> <img src='images/style/kandinsky2.jpg' height='192'> </p><p><img src='images/content/gbk.jpg' height='192'> <img src='res/gbk_kandinsky.jpg' height='192'> <img src='images/style/kandinsky.jpg' height='192'> </p><p><img src='images/content/gbk.jpg' height='192'> <img src='res/gbk_picasso.jpg' height='192'> <img src='images/style/picasso.jpg' height='192'> </p><p><img src='images/content/gbk.jpg' height='192'> <img src='res/gbk_the_scream.jpg' height='192'> <img src='images/style/the_scream.jpg' height='192'> </p><p><img src='images/content/gbk.jpg' height='192'> <img src='res/gbk_monet.jpg' height='192'> <img src='images/style/monet.jpg' height='192'> </p><p><img src='images/content/gbk.jpg' height='192'> <img src='res/gbk_seated-nude.jpg' height='192'> <img src='images/style/seated-nude.jpg' height='192'> </p><p><img src='images/content/gbk.jpg' height='192'> <img src='res/gbk_shipwreck.jpg' height='192'> <img src='images/style/shipwreck.jpg' height='192'> </p><p><img src='images/content/gbk.jpg' height='192'> <img src='res/gbk_woman-with-hat-matisse.jpg' height='192'> <img src='images/style/woman-with-hat-matisse.jpg' height='192'> </p><h3>tubingen.jpg</h3><p><img src='images/content/tubingen.jpg' height='192'> <img src='res/tubingen_starry-night.jpg' height='192'> <img src='images/style/starry-night.jpg' height='192'> </p><p><img src='images/content/tubingen.jpg' height='192'> <img src='res/tubingen_kandinsky2.jpg' height='192'> <img src='images/style/kandinsky2.jpg' height='192'> </p><p><img src='images/content/tubingen.jpg' height='192'> <img src='res/tubingen_kandinsky.jpg' height='192'> <img src='images/style/kandinsky.jpg' height='192'> </p><p><img src='images/content/tubingen.jpg' height='192'> <img src='res/tubingen_picasso.jpg' height='192'> <img src='images/style/picasso.jpg' height='192'> </p><p><img src='images/content/tubingen.jpg' height='192'> <img src='res/tubingen_the_scream.jpg' height='192'> <img src='images/style/the_scream.jpg' height='192'> </p><p><img src='images/content/tubingen.jpg' height='192'> <img src='res/tubingen_monet.jpg' height='192'> <img src='images/style/monet.jpg' height='192'> </p><p><img src='images/content/tubingen.jpg' height='192'> <img src='res/tubingen_seated-nude.jpg' height='192'> <img src='images/style/seated-nude.jpg' height='192'> </p><p><img src='images/content/tubingen.jpg' height='192'> <img src='res/tubingen_shipwreck.jpg' height='192'> <img src='images/style/shipwreck.jpg' height='192'> </p><p><img src='images/content/tubingen.jpg' height='192'> <img src='res/tubingen_woman-with-hat-matisse.jpg' height='192'> <img src='images/style/woman-with-hat-matisse.jpg' height='192'> </p>

## Main references

* Original paper
* https://github.com/hwalsuklee/tensorflow-style-transfer
* And more ++