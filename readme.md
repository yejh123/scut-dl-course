# SCUT Deep Learning Course 
该github项目旨在以初学者的角度，帮助同学们更好的理解深度学习的实现算法，把训练过程的代码框架提供给同学们，省去同学们Debug的时间，因为只提供代码的框架，所以具体的模型实现代码会被删除。

深度学习并没有大家想象的那么困难，本质和搭积木没什么区别，无非就是把积木换成了一些常见的模型层（Layer）。所以，想要学好Deep Learning，只需要掌握好理论知识+花一点时间动手实践就好了。

项目使用的深度学习框架为PyTorch，如果您是TensorFlow的使用者，请参考：

- [【中文教程 6.3k star】tensorflow2_tutorials_chinese](https://github.com/czy36mengfei/tensorflow2_tutorials_chinese)
- [TensorFlow-Examples 39k star](https://github.com/aymericdamien/TensorFlow-Examples)

当然，如果想要参考其他的PyTorch实现：
- [pytorch-tutorial 18.3k star](https://github.com/yunjey/pytorch-tutorial)
- [examples 14.3k star](https://github.com/pytorch/examples)


## 储备知识
在正式开始编写代码之前，请确认自己知道一下概念：

 - 训练集和测试集的区别和各自的作用范围
 - 随机梯度下降算法 (SGD) （很重要，实际的训练必须依靠SGD）
 - 前向传播(forward)和反向传播(backward)的基本原理
 - 常见的几种激活函数：Sigmoid、Tanh、Relu
 - 常见的学习器 (optimizer)：Adam、RMSProp
 - Feedforward Network (hw1)
 - Convolution Neural Network（hw2）: Conv2d、MaxPooling
 - Recurrent Neural Network (hw3)
 - AutoEncoder (hw?)
 - Generative Adversarial Network (hw?)
 
 
如果有不清楚的地方，请自行Google（需科学上网）/知乎（推荐）/Baidu（不建议），或者与我或者助教交流。


## 全连接神经网络 (DNN)


## Back Propogation算法
深入理解δ学习规则，见PPT


## 卷积神经网络 (CNN)
为什么CNN在处理图像问题上这么强大，因为CNN利用了许多图片的相关特性。图片的特性有：

| 特性 | 图片 | CNN |
|  ----  | ---- | ---- |
| 局部性 | 像素与像素之间，主要与其附近的像素有关，而与其距离较远的像素无关 | 与DNN不同，CNN采用了稀疏连接的方法，一次提取出一个Filter大小的数据 |
| 统计平稳性 | 像素的统计性指标在整幅图像中是相对统一的 | 所以一个Filter可以对整个图像进行多次特征提取，从而实现权重共享 |
| 平移不变性 | 对于物体的识别不依赖于它在图像中的位置 | 是MaxPool的前提假设 |
| 构成性 | 被识别目标是由各个部分构成的 | 随着层数的加深，一个Filter提取的图像范围加大 |

- [一文看懂卷积神经网络-CNN（基本原理+独特价值+实际应用）](https://medium.com/@pkqiang49/%E4%B8%80%E6%96%87%E7%9C%8B%E6%87%82%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-cnn-%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86-%E7%8B%AC%E7%89%B9%E4%BB%B7%E5%80%BC-%E5%AE%9E%E9%99%85%E5%BA%94%E7%94%A8-6047fb2add35)

### Conv2d

![卷积核](https://miro.medium.com/max/658/0*sxpi42l2IIpS2vuJ.gif)

### MaxPool2d

![池化层](https://miro.medium.com/max/875/0*JjtDg7FAhjQOrld-.gif)

### ConvTranspose2d (卷积转置)

 - [ConvTranspose2d原理，深度网络如何进行上采样？](https://blog.csdn.net/qq_27261889/article/details/86304061



### 扩展：如何使得CNN更强大
 - 修改模型架构：又深又宽，招人喜欢（误），一般来说，模型深>模型宽（详见相关论文），前提是你跑的起来，而且具有避免梯度消失/梯度爆炸的方法（Residual Network）
 - 激活函数：改用Relu激活函数防止梯度消失
 - 正则化
  - Batch Normalization：对输入数据标准化
  - Dropout：一种正则化方法（不建议与Batch Normalization一起使用，[PyTorch 51.BatchNorm和Dropout层的不协调现象](https://zhuanlan.zhihu.com/p/199521441)）
 - 图像增强 (Image Augmentation)：一张图片经过裁剪、伸缩、旋转可以变化出上百张图片，从而大大加大了数据集规模
 - 集成&投票 (Ensemble&Voting)：训练多个模型，让这些模型都对同一种图片进行分类，最终结果投票决定
 - 终极大招，又称站在巨人的肩膀上之法：Fine-tune，直接调用AlexNet、VGGNet、ResNet、Inception等已经pre-trained好的模型进行调整


## 循环神经网络 (RNN)

RNNs因为具有**记忆**功能，所以主要用于处理序列化数据，例如自然语言、视频、语音等数据，是深度学习中最常用的模型之一。

### BPTT算法

## LSTM
![LSTM公式](https://github.com/yejh123/scut-dl-course/blob/main/RNN/LSTM.png)

![双向LSTM示意图](https://github.com/yejh123/scut-dl-course/blob/main/RNN/double%20LSTM.jpg)

## GRU
![GRU公式](https://github.com/yejh123/scut-dl-course/blob/main/RNN/GRU.png)

## Seq2Seq 
序列化到序列化模型，即输入一段序列化数据，输出另一端序列化数据，是深度学习的一大研究方向，广泛应用于NLP等领域。

 - [yejh123/Seq2Seq-Translation-GRU-Attention](https://github.com/yejh123/Seq2Seq-Translation-GRU-Attention/tree/main)

## Transformer
Transformer是对传统RNNs模型的一大改进，模型应用了符合人类认知直觉的**Attention**机制。

Transformer的出现（2017年），使得NLP领域近年来突飞猛进，出现了许多著名且强大的语言模型，包括BERT、GPT-2、GPT-3等。

 - [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 
 - [直觀理解GPT-2 語言模型並生成金庸武俠小說 - LeeMeng](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html)
 - [yejh/Transformer](https://github.com/yejh123/Transformer)
 
## Auto-Encoder
 - 普通Auto-Encoder
 - 正则Auto-Encoder
   - 稀疏Auto-Encoder：L1正则化
   - 噪声Auto-Encoder


## hw1

### hw1-1
numpy基本操作

### hw1-2
一维线性回归

### hw1-3
多项式线性回归

### hw1-4
使用线性回归进行MNIST分类


## hw2

### hw2-1
使用线性回归进行MNIST分类

### hw2-2
使用线性回归进行MNIST分类

### hw2-3
手动实现BP算法


## hw3


## hw4









