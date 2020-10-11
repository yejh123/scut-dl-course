# CNN论文集学习笔记
## ILSVRC
是一个比赛，全称是ImageNet Large-Scale Visual Recognition Challenge，平常说的ImageNet比赛指的是这个比赛。

使用的数据集是ImageNet数据集的一个子集，一般说的ImageNet（数据集）实际上指的是ImageNet的这个子集，总共有1000类，每类大约有1000张图像。具体地，有大约1.2 million的训练集，5万验证集，15万测试集。

ILSVRC从2010年开始举办，到2017年是最后一届。ILSVRC-2012的数据集被用在2012-2014年的挑战赛中（VGG论文中提到）。ILSVRC-2010是唯一提供了test set的一年。

ImageNet可能是指整个数据集（15 million），也可能指比赛用的那个子集（1000类，大约每类1000张），也可能指ILSVRC这个比赛。需要根据语境自行判断。

12-15年期间在ImageNet比赛上提出了一些经典网络，比如AlexNet，ZFNet,OverFeat，VGG，Inception，ResNet。我在CNN经典结构1中做了相应介绍。

16年之后也有一些经典网络，比如WideResNet，FractalNet，DenseNet，ResNeXt，DPN，SENet。我在CNN经典结构2中做了相应介绍。



## AlexNet (2012)
论文：ImageNet Classification with Deep Convolutional Neural Networks

 - Top-1正确率36.7%，Top-5正确率15.32%。
 - 7层
 - 60 million parameters

### Data Augmentation
输入图像处理，在训练中使用Data Augmentation，可以使训练数据更多，提高模型准确率；在预测时使用Data Augmentation，可以获得更多的图片，对于每一张图片，输入到模型中会得到一个结果，将所有结果求平均，得到最终的结果。

### Dropout=0.5
提供模型的正则化作用。

### Local Response Normalisation
局部响应归一化，该技术逐渐被Batch Normalization取代。



## VGGNet (2014亚军)
论文：VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION

 - 单个模型（不做ensemble）Top-1正确率24.4%，Top-5正确率7.0%。
 - 最多有19层

### Multi-Scale Training
多尺度训练，输入图片的维度不确定，落在一个区间范围。

Top-1正确率24.8%，Top-5正确率7.5%。

### Mutil-Crop Valuation（多裁剪验证）
将每张图片转化成三张不同维度的图片，每张图片Flip后裁剪出50个224*224的图片，共获得150个224*224的图片。

将150个224*244图片丢入模型进行验证，对得到的150的结果求平均，得到最终结果。

Top-1正确率24.4%，Top-5正确率7.1%。

### Ensemble
将16层VGGNet与19层VGGNet作ensemble。

Top-1正确率23.7%，Top-5正确率6.8%。




## GoogleNet/Inception (2014冠军)
论文：Going deeper with convolutions

 - 22层但是参数较少，只有AlexNet的1/12
 - Top-5正确率6.67%

### 1*1卷积
 - 增加非线性，有利于提取更加抽象的深层特征
 - 特征降维，减少计算量和参数数量

参数减少不一定是坏事，反而参数太多，模型很难训练，并且容易过拟合。


### Auxiliary Classification
在训练阶段加入辅助分类器，避免梯度消失问题，并且使得从模型中层提取到的特征就具有更好的区分度，能够根据这些特征就预测图片类型。


### Local Response Normalisation
方法出自AlexNet (2012)，但此方法不一定适用于所有CNN模型（VGGNet论文提出LRN对本模型没有帮助）。


### Ensemble
训练了7个模型，各自的训练数据的打乱顺序不一样。


### Data Augmentation/Crops
把每张图片按比例调整为短边分别为256、288、320、352的4个不同维度图片。

对4个图片分别提取左中右3个不同位置的图片。

对3个图片在四个角和中心的五个位置进行裁剪，每个图片可以得到5个裁剪后的图片和1个裁剪前的图片。

对6个图片（5个裁剪过和1个未裁剪）进行Resize，维度为224*224。

对6个图片进行镜像翻转，又得到6个新的图片。

所以，一张图片经过Data Augmentation后能得到4*3*6*2=144张不同的图片。

把144张图片输入到模型中，得到144个结果，对所有结果求平均，得到最终分类。


### 改进：Rethinking the Inception Architecture for Computer Vision
关于Inception的第二篇论文主要对Inception module做了更改，新增了几个inception modules，并使用了Batch Normalization，升级得到了新的Inception模型Inception-v2和Inception-v3。

#### 卷积分解
m*n的卷积大小可以分解为m*1和1*n两个卷积。


#### 标签平滑
将独热编码（只有1项为1，其他项全为0）修改为全非零编码，使得每个样本数据是其他类别的概率都不为0。

 - 有一定的正则化作用



## Batch Normalization (2015)
论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

[Pytorch batch_normalization层详解](https://blog.csdn.net/winycg/article/details/88974107)

### Internal Covariate Shift

### 好处
 - 使用了BN的模型可以使用更高的learning rate，加速模型学习速度
 - BN使得模型对初始化方法和网络中的参数不那么敏感，简化调参过程，使得网络学习更加稳定
 - BN允许网络使用饱和性激活函数（例如sigmoid，tanh等），缓解梯度消失问题
 - BN有一定的正则化作用，因此可以移除L2正则化的使用、Dropout、Data Augmentation
 - 对于不同Batch中同样的图片，因为不同的Batch的平均值和方差不同，因此会有一定的打乱数据的作用，提高模型泛化能力
 



## ResNet (2015)
论文：Deep Residual Learning for Image Recognition

 - 152层
 - Top-5正确率3.57%
 - 计算量比VGGNet少

### Residual Learning
假设输入为x，输出为y

F(x)是x先后经过卷积层、BN层、ReLU激活函数、卷积层、BN后的输出。

对于F(x)与x维度相同的情况，输出y = ReLU(F(X) + x)

F(X)可以看做是高层特征，x可以看做是底层特征。

如果F(x)与x维度不同，下面介绍两种方法：

#### Zero-padding
按步长在多余的空位补0。正确率不如Projection Shortcut方法。


#### Projection Shortcut
使用1*1的卷积层，设置合理的步长和特征图数量，使得F(x)与x维度相同。

采用Projection Shortcut，正确率比Zero-padding高。

如果在所有的Residual层都采用Projection Shortcut，正确率比只在F(x)和x维度不同的时候使用Projection Shortcut还要高。


### Traning
输入图片维度在[256, 480]之间。

对图片做水平翻转，每个像素减去像素平均值。

随机裁剪出224*224的crop，作为模型输入。

在每一卷积层后面、激活函数前面，加上BN，不使用Dropout。

初始化模型参数。

使用SGD，每个mini-batch为256进行训练。

learning rate从0.1开始，并且当错误率难以下降时，将learning rate除以10.

weight decay=0.0001, momentum=0.9

共训练600k次。

### Testing
Resize输入图片到{224, 256, 384, 480, 640}四个维度，每个不同维度图片作10-crop testing。


### 改进：Identity Mappings in Deep Residual Networks
Identity Mappings in Deep Residual Networks是关于ResNet的第二篇论文，主要对残差结构做了调整。

假设输入为x，输出为y

F(x)是x先后经过BN层、ReLU激活函数、卷积层、BN、ReLU激活函数、卷积层后的输出。

对于F(x)与x维度相同的情况，输出y = F(X) + x



## Inception-ResNet
论文：Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning

 - Top-5正确率3.08%
 - 融合了Inception的结构和ResNet的残差网络








