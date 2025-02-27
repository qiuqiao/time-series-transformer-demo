# 多层因果Transformer

## 简介

非常简单的小项目，仅仅用于完成学校的任务

这是一个基于PyTorch Lightning实现的用于自回归时间序列预测的Transformer模型。输入输出均为(B,T,C)，B为批次大小，T为序列长度，C为特征数量，每一个时间步都用于预测下一个时间步的结果。

例：

```python
y = model(x)
```

x的维度为(B,T,C)，y的维度为(B,T,C)，y[b,t,c] 是使用 x[b,0:t,:]进行预测的。

## 特点

- 多层Transformer架构
- 因果注意力机制，确保每个时间步只能看到当前及之前的信息
- 支持多头注意力
- 包含位置编码，捕获序列中的位置信息
- 可配置的模型大小（层数、隐藏维度、注意力头数等）
- 基于PyTorch Lightning的训练框架
- 支持学习率自动调整
- 支持早停和模型检查点
- 支持TensorBoard可视化
- 自动滑动窗口数据处理
- 多特征预测支持
- 预测结果可视化

## 安装

1. 克隆仓库：

```bash
git clone <repository-url>
cd mytransformer
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

依赖版本要求：
- torch >= 1.9.0
- numpy >= 1.20.0
- matplotlib >= 3.4.0
- pytest >= 6.2.5
- pytorch-lightning >= 2.0.0
- tensorboard >= 2.10.0

## 使用方法

### 基本用法

```python
import torch
from transformer import CausalTransformer

# 初始化模型
model = CausalTransformer(
    input_dim=10,      # 输入特征维度
    hidden_dim=64,     # 隐藏层维度
    output_dim=10,     # 输出特征维度
    num_layers=3,      # Transformer层数
    num_heads=4,       # 注意力头数
    ff_dim=256,       # 前馈网络维度（可选）
    dropout=0.1       # Dropout率（可选）
)

# 创建输入数据
batch_size = 32
seq_len = 24
input_dim = 10
x = torch.randn(batch_size, seq_len, input_dim)

# 前向传播
y = model(x)  # 输出形状: (batch_size, seq_len, output_dim)
```

### CSV数据支持

模型支持使用CSV数据文件进行训练。CSV中的每一行代表一个时间步，每一列代表一个特征。特征列应命名为"feature_1"、"feature_2"等。

1. 生成一个玩具数据集：

```bash
python generate_csv_data.py
```

这将在`data`目录中创建训练集、验证集和测试集。

2. 使用CSV数据训练模型：

先在`data`目录中准备好`train_data.csv`和`val_data.csv`，然后运行：

```bash
python train.py
```

训练参数可在train.py中修改：
- seq_len: 输入序列长度（默认100）
- pred_len: 预测序列长度（默认20）
- batch_size: 批次大小（默认32）
- max_epochs: 最大训练轮数（默认30）
- learning_rate: 学习率（默认0.001）
- hidden_dim: 隐藏层维度
- num_layers: Transformer层数
- num_heads: 注意力头数

可以使用tensorboard可视化训练过程：

```bash
tensorboard --logdir=lightning_logs
```

### 预测和可视化

训练完成后，可以使用模型进行预测：

先在`data`目录中准备好`test_data.csv`，然后运行：
```bash
python predict.py [参数]
```

预测脚本支持以下参数：
- `--model_path`：模型检查点路径（可选，不指定则自动选择验证损失最低的检查点）
- `--checkpoints_dir`：检查点目录路径（默认："checkpoints"）
- `--data_path`：测试数据CSV文件路径（默认："data/test_data.csv"）
- `--seq_len`：输入序列长度（默认：100）
- `--pred_len`：预测序列长度（默认：20）
- `--output_dir`：输出目录（默认："predictions"）

预测结果的值会保存在`output_dir/prediction_results.csv`中，预测结果图表会保存在`output_dir/prediction_result_feature_{i}.png`中，i为特征索引。

示例：
```bash
# 使用默认参数
python predict.py

# 指定参数
python predict.py \
    --model_path checkpoints/best_model.ckpt \
    --data_path data/my_test_data.csv \
    --seq_len 150 \
    --pred_len 30 \
    --output_dir my_predictions
```

预测过程说明：
1. 模型采用自回归（autoregressive）方式生成预测序列
2. 输入序列会被限制在 `seq_len` 长度内（过长则裁剪），以确保与训练时的输入长度匹配
3. 每次预测时会使用之前的预测结果作为下一步预测的输入

预测结果输出：
1. 预测结果图表（每个特征一个图表）：
   - 文件名：`prediction_result_feature_{i}.png`
   - 包含真实值（蓝色实线）
   - 预测值（红色虚线）
   - 预测起始点标记（绿色竖线）

2. 预测结果数据：
   - 文件名：`prediction_results.csv`
   - 包含时间步、每个特征的真实值和预测值

## 模型结构

- **输入投影**：将输入特征映射到模型的隐藏维度
- **位置编码**：添加位置信息
- **Transformer层**：多个Transformer编码器层，每层包含：
  - 多头自注意力机制（带有因果掩码）
  - 层归一化
  - 前馈神经网络
- **输出投影**：将隐藏表示映射回输出特征维度

## 单元测试

运行单元测试以验证模型的正确性：

```bash
pytest test_transformer.py
```

测试包括：
- 输出形状测试
- 因果掩码测试
- 自回归属性测试
- 梯度流测试
- 多步预测测试
- 批次独立性测试
- 模型组件测试


