# MachineLearing
[TOC]
machine learing in action

I will pull the code, which is checked out as correct, and I'll updated continuously.

Please stay tuned


- fix-B

Feature-A


## 1、决策树
- 决策树原理就是通过特征选择的方法递归生成决策树，最终通过决策树剪枝减少一些增益比较小的子树。
- 特征选择
  - ID3：以信息增益为特征选择准则
  - C4.5：以信息增益比为特征选择准则
  - CART：以基尼指数为特征选择标准
- 决策树的生成
  - 递归地进行特征选择，生成决策树结构
- 决策树剪枝
  - 剪取信息增益较小的子树，避免过拟合

## 2、LightGBM提升学习模型
- 原理和XGBoost一样，都是通过使用残差的泰勒展开式近似代替残差，另外通过给损失函数添加正则化项，降低了模型的复杂度，避免过拟合。
- 利用leaf-wise分裂策略代替XGBoost的level-wise分裂策略，减去增益较小的结点的分裂，大大增加了模型的训练速度。
- 利用直方图算法代替XGBoost中的exact中使用的预排序算法，只需要保存特征离散化后的值，省去了保存原始特征所有值的过程，减少了内存的使用，并加速了模型的训练速度。