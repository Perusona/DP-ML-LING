# 机器学习&深度学习笔记（LING自用）

- 实验环境：

  - 系统：Win11
  - IDE：VSCode
- 拉取请使用code:

  ```git
  git clone https://github.com/Perusona/DP-ML-LING.git
  ```

## 目录

1. 计算机视觉
2. 机器学习
   1. 全连接神经网络（FCN-Net）
      1. 独热编码（one-hot encoding）
3. 深度学习
4. 数据挖掘

## FCN-Net

### 独热编码

举例来说，数据集中有$N$个样本，分别为$x^1,x^2,x^3,…,x^N$。每一个样本都有$n$个特征值，对于数据集中的第$i$个样本，其$n$个特征值分别为$x_1^i,x_2^i,x_3^i,…,x_n^i$。所有样本的标签分为$3$类，分别为$c_1 、c_2 、c_3$，可以把这$3$类标签分别表示为$0、1、2$。这样每一个样本的标签都为$0、1、2$中的一个值。

- [VScode快捷键速查表](https://www.cheat-sheet.cn/post/vs-code-keyboard-shortcuts/#:~:text=Visual%20Studio%20Code%E5%BF%AB%E6%8D%B7%E9%94%AE%E9%80%9F%E6%9F%A5%E8%A1%A8)
