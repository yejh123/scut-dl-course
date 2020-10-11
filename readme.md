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

### Conv2d
构成CNN的模块之一

### MaxPool2d
构成CNN的模块之二

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

### BPTT算法

### LSTM

### GRU


## Transformer
大人，时代变了！



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









