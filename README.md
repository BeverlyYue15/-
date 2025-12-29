# 集成学习回归预测模型

基于 Scikit-learn 的集成学习回归预测项目，使用多种机器学习模型对治疗效果进行预测。

## 项目简介

本项目使用集成学习方法，结合 Gradient Boosting、Random Forest 和 AdaBoost 三种回归模型，对患者治疗后的指标总分进行预测。通过网格搜索进行超参数调优，并使用加权平均方式融合多个模型的预测结果。

## 数据说明

输入数据为 Excel 文件，包含两个 Sheet：
- **Sheet1**: 训练集
- **Sheet2**: 测试集

数据结构：
| 列索引 | 说明 |
|--------|------|
| 0-29 | 治疗前的30项指标分数（数值型特征） |
| 30 | 所服药物（分类型特征） |
| 31 | 治疗后的指标总分（目标变量） |

## 技术栈

- Python 3.x
- pandas - 数据处理
- numpy - 数值计算
- scikit-learn - 机器学习模型
- openpyxl - Excel 文件读取

## 模型架构

### 基学习器
1. **Gradient Boosting Regressor** - 梯度提升回归
2. **Random Forest Regressor** - 随机森林回归
3. **AdaBoost Regressor** - 自适应提升回归

### 超参数调优
使用 5 折交叉验证的网格搜索（GridSearchCV）对每个模型进行超参数优化：

- **Gradient Boosting**: n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf
- **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf
- **AdaBoost**: n_estimators, learning_rate, loss

### 模型融合
采用简单平均法融合三个最优模型的预测结果：
```
y_pred = (gb_pred + rf_pred + ada_pred) / 3.0
```

## 安装依赖

```bash
pip install pandas numpy scikit-learn openpyxl
```

## 使用方法

1. 将数据文件放置在 `附件1/回归预测.xlsx` 路径下
2. 运行脚本：

```bash
python sheet1.py
```

## 输出结果

程序将输出以下内容：
- 各模型的最优超参数
- 交叉验证得分（MSE）
- 测试集平方相对误差（Squared Relative Error）统计：
  - 均值
  - 方差
  - 标准差
  - 最小值/最大值
  - 中位数
- 前10个样本的详细预测结果

## 评估指标

采用**平方相对误差（Squared Relative Error）**作为评估指标：

$$SRE = \left(\frac{\hat{y} - y}{y}\right)^2$$

其中 $\hat{y}$ 为预测值，$y$ 为真实值。

## 项目结构

```
.
├── README.md
├── sheet1.py          # 主程序
└── 附件1/
    └── 回归预测.xlsx   # 数据文件
```

## License

MIT License
