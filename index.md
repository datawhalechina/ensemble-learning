## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/datawhalechina/ensemble-learning/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

﻿# 开源项目：机器学习集成学习与模型融合(基于python)

## (一) 项目初衷

这件事开始于我们参加一些机器学习比赛，在类似于kaggle等大型数据科学竞赛的时候, 我们总喜欢去观摩高分选手的一些参赛模型，得出一个很重要的结论就是：除了深度学习以外的高分模型，无一例外地见到了集成学习和模型融合的身影。这个发现迫使我去学习一些除了基础模型以外的集成学习方法以便在这些比赛上获得更好的成绩。但是，当我使用具体的sklearn编程的时候, 往往因为不懂得集成学习的一 些底层知识而不懂参数的含义。因此，在本项目中我们会从基础模型的推导以及 sklearn应用过渡到使用集成学习的技术去优化我们的基础模型，使得我们的模型能更好地解决机器学习问题。

## (二) 内容设置

- 第一章：机器学习数学基础（待完善）

  - 高等数学微分学

  - 线性代数

  - 概率论与数理统计

  - 随机过程与抽样原理

- 第二章：机器学习基础
  - 机器学习的三大主要任务
  - 基本的回归模型
  - 偏差与方差理论
  - 回归模型的评估及超参数调优
  - 基本的分类模型
  - 分类问题的评估及超参数调优
- 第三章：集成学习之投票法与Bagging
  - 投票法的思路
  - 投票法的原理分析
  - 投票法的案例分析(基于sklearn，介绍pipe管道的使用以及voting的使用)
  - Bagging的思路
  - Bagging的原理分析
  - Bagging的案例分析(基于sklearn，介绍随机森林的相关理论以及实例)
- 第四章：集成学习之Boosting提升法
  - Boosting的思路与Adaboost算法
  - 前向分步算法与梯度提升决策树(GBDT)
  - XGBoost算法与xgboost库的使用
  - Xgboost算法案例与调参实例
  - LightGBM算法的基本介绍
- 第五章：集成学习之Blending与Stacking
  - Blending集成学习算法
  - Stacking集成学习算法
  - Blending集成学习算法与Stacking集成学习算法的案例分享
- 第六章：集成学习之案例分析
  - 集成学习案例一 （幸福感预测）
  - 集成学习案例二 （蒸汽量预测）

## (三) 人员安排

| 成员   | 个人简介                                              | 个人主页                                          |
| ------ | ----------------------------------------------------- | ------------------------------------------------- |
| 萌弟 | Datawhale成员，项目负责人，深圳大学数学与应用数学专业，机器学习算法工程师 | https://www.zhihu.com/people/meng-di-76-92/posts  |
| 六一   | Datawhale成员，算法工程师                     |                                                   |
| 杨毅远 | Datawhale成员，清华大学自动化系研二               | https://yyysjz1997.github.io/                     |
| 薛传雨 | Datawhale成员，康涅狄格大学在读博士                   | http://chuanyuxue.com/                            |
| 陈琰钰 | Datawhale成员，清华大学深圳研究生院研一               | https://cyy0214.github.io/                        |
| 李嘉骐 | 清华大学自动化系在读博士                              | https://www.zhihu.com/people/li-jia-qi-16-9/posts |

教程贡献情况：

李祖贤： CH1、CH2、CH4、CH5

薛传雨：CH3

杨毅远：CH6

李嘉骐：CH3优化

组队学习贡献情况：

六一：长期学习流程设计、组织协调、23期运营&作业评审（task1）、24期运营&作业评审（task3）

萌弟：23期答疑&直播（3次）&作业评审(task3&4)、24期答疑&直播（3次）&作业评审(task4&5)

薛传雨：23期运营&答疑&作业评审（task5）、24期运营

陈琰钰：23期作业评审（task2&6）

杨毅远：23期答疑

李嘉骐：24期答疑&作业评审（task1&2）

(四) 课程编排与使用方法

- 课程编排：
  课程现分为三个阶段，大致可以分为：机器学习模型回顾，集成学习方法的进阶, 集成学习项目的实践。
  1. 第一部分：当我们可以比较熟练的操作数据并认识这个数据之后，我们需要开始数据清洗以及重
     构, 将原始数据变为一个可用好用的数据, 基于sklearn构建模型以及模型评价，在这个部分我们会重点详细学习各个基础模型的原理以及sklearn的各个参数。
  2. 第二部分：我们要使用sklearn, xgboost, lightgbm以及mIxtend库去学习集成学习的具体方法以及原理底层。
  3. 第三单元：通过前面的理论学习，现在可以开始进行实践了，这里有两个大型集成学习项目的实践。
- 使用方法:
  我们的代码都是jupyter notebook和markdown形式, 我们在每一章后面会给出几道小习题方便大家掌握。其中的内容会作为组队学习的项目安排!

## (五) 反馈

- 如果有任何想法可以联系邮箱 (1028851587@qq.com)
- 如果有任何想法可以联系我们DataWhale
