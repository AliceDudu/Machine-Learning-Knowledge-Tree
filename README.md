# Machine-Learning-Knowledge-Tree

今天开始会陆续将之前的文章做一下梳理，整理出一个完整的知识体系，有需要的伙伴们可以更方便地查找自己需要的知识点。

## 【】了解机器学习问题一般包括哪几步：

![](https://upload-images.jianshu.io/upload_images/1667471-b58174aea587575f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这里提到的三篇文章比较全地展示了机器学习问题的一般流程：
---

### 1. [一个框架解决几乎所有机器学习问题](https://github.com/AliceDudu/Machine-Learning-Knowledge-Tree/blob/master/001-%E4%B8%80%E8%88%AC%E6%B5%81%E7%A8%8B/%E4%B8%80%E4%B8%AA%E6%A1%86%E6%9E%B6%E8%A7%A3%E5%86%B3%E5%87%A0%E4%B9%8E%E6%89%80%E6%9C%89%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E9%97%AE%E9%A2%98.md)

这篇文章介绍了应用算法**解决 Kaggle 问题，一般有以下几个步骤，**
以及每个步骤的简要定义和常用方法：

>第一步：识别问题
第二步：分离数据
第三步：构造提取特征
第四步：组合数据
第五步：分解
第六步：选择特征
第七步：选择算法进行训练

这篇文章中的**关键知识点：**

- 为什么需要将数据分成 train/valid/test 部分？
- 每种模型需要调节什么参数？
- 179种分类模型在UCI所有的121个数据上的性能比较
- 训练集 & 测试集应用模型的流程有什么区别？

---

### 2. [通过一个kaggle实例学习解决机器学习问题](https://github.com/AliceDudu/Machine-Learning-Knowledge-Tree/blob/master/001-%E4%B8%80%E8%88%AC%E6%B5%81%E7%A8%8B/%E9%80%9A%E8%BF%87%E4%B8%80%E4%B8%AAkaggle%E5%AE%9E%E4%BE%8B%E5%AD%A6%E4%B9%A0%E8%A7%A3%E5%86%B3%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E9%97%AE%E9%A2%98.md)

这篇文章**用一个实例来将上一篇的流程应用了一下：**

>Data Exploration
Data Cleaning
Feature Engineering
Model Building
Ensemble Learning
Predict

**以及建立 训练 并用模型预测的过程：**

>从 sklearn 导入分类器模型后，定义一个 KNN，
定义合适的参数集 parameters，
然后用 get_model 去训练 KNN 模型，
接下来用训练好的模型去预测测试集的数据，并得到 accuracy_score，
然后画出 learning_curve。

这篇文章中的**关键知识点：**

- 分类问题的常用数据探索方法
- 缺失值如何处理？
- 如何通过原始变量构造新的特征？

---

### 3. [从 0 到 1 走进 Kaggle](https://github.com/AliceDudu/Machine-Learning-Knowledge-Tree/blob/master/001-%E4%B8%80%E8%88%AC%E6%B5%81%E7%A8%8B/%E4%BB%8E%200%20%E5%88%B0%201%20%E8%B5%B0%E8%BF%9B%20Kaggle.md)

这篇文章介绍了 Kaggle 比赛的一般流程：

>探索数据
特征工程
建立模型
调参
预测提交

文章中的**关键知识点：**

- 如何探索数据？
- 如何构造特征？
