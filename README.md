# 九、集成学习回归预测模型

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


# 十、CVRPTW Solver

基于启发式组合优化算法的带容量和时间窗约束的车辆路径问题(CVRPTW)求解器。

## 问题描述

CVRPTW (Capacitated Vehicle Routing Problem with Time Windows) 是经典的组合优化问题：

- **目标**：最小化车辆行驶总距离
- **约束**：
  - 每辆车有容量限制
  - 每个客户有时间窗限制（最早/最晚服务时间）
  - 每个客户只能被访问一次
  - 所有路径从仓库出发并返回仓库

## 算法设计

本求解器采用三阶段混合启发式算法：

### 1. 构造启发式 - Solomon I1 插入法

- 选择种子客户策略：最远距离 / 最早截止时间
- 按最小插入成本迭代插入未路由客户
- 同时考虑距离增量和时间窗紧迫性

### 2. 局部搜索改进

| 算子 | 描述 |
|------|------|
| **2-opt** | 路径内反转一段客户序列 |
| **Or-opt** | 移动连续1-3个客户到其他位置 |
| **Relocate** | 将客户从一条路径移动到另一条 |
| **Exchange** | 交换两条路径间的客户 |

### 3. 元启发式 - 模拟退火 (Simulated Annealing)

- 初始温度：100
- 冷却率：0.997
- 迭代次数：10000
- 定期局部搜索强化

## 项目结构

```
.
├── cvrptw_solver.py    # 主求解器代码
├── run.sh              # 运行脚本
├── README.md           # 项目说明
└── 附件2/
    ├── data.txt        # 测试数据
    └── 数据说明.pdf     # 数据格式说明
```

## 数据格式

```
VEHICLE
NUMBER     CAPACITY
  10         150

CUSTOMER
CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE
    0      40         50          0          0       1236
    1      45         68         10        912        967
    ...
```

- `CUST NO.`: 客户编号（0为仓库）
- `XCOORD/YCOORD`: 坐标
- `DEMAND`: 需求量
- `READY TIME`: 时间窗开始
- `DUE DATE`: 时间窗结束

距离计算：欧式距离取整 `int(sqrt((x1-x2)² + (y1-y2)²))`

## 使用方法

### 环境要求

- Python 3.6+
- 无需额外依赖包

### 运行

```bash
# 方式1：直接运行
python3 cvrptw_solver.py

# 方式2：使用脚本
./run.sh

# 方式3：指定数据文件
python3 cvrptw_solver.py path/to/data.txt
```

## 求解结果

测试数据（50客户，10车辆，容量150）：

| 阶段 | 总距离 |
|------|--------|
| 初始解 (Solomon) | 724 |
| 局部搜索后 | 600 |
| 模拟退火后 | **469** |

**最优解详情：**

| 车辆 | 路径 | 载重 | 距离 |
|------|------|------|------|
| 0 | 0→5→3→7→8→10→11→9→4→2→1→0 | 140 | 56 |
| 1 | 0→20→24→25→27→29→30→28→26→22→0 | 140 | 48 |
| 2 | 0→43→42→41→40→44→46→45→48→50→49→47→0 | 140 | 57 |
| 3 | 0→13→17→18→19→15→23→21→0 | 150 | 86 |
| 4 | 0→32→33→38→39→36→34→0 | 150 | 86 |
| 5 | 0→31→35→37→16→14→12→6→0 | 140 | 136 |

- 使用车辆：6/10
- 总距离：469

## 输出格式

```
Route for vehicle 0:
 0 Load(0) Time(0)-> 5 Load(10) Time(15)-> ... -> 0 Load(140) Time(930)
Distance of the route: 56
```

- `Load(x)`: 累计载重
- `Time(x)`: 到达时间

## 算法复杂度

- 构造启发式：O(n³)
- 局部搜索：O(n² × iterations)
- 模拟退火：O(n × max_iter)

## 扩展

可通过修改以下参数优化求解效果：

```python
# 模拟退火参数
initial_temp = 100.0    # 初始温度
cooling_rate = 0.997    # 冷却率
max_iter = 10000        # 最大迭代次数

# 局部搜索迭代次数
local_search_iter = 500
```

## License

MIT License
