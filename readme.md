# 多层因果Transformer

## 简介

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

1. 生成样本CSV数据：

```bash
python generate_csv_data.py
```

这将在`data`目录中创建训练集、验证集和测试集。

2. 使用CSV数据训练模型：

```bash
python train.py
```

训练参数可在train.py中配置：
- seq_len: 输入序列长度（默认100）
- pred_len: 预测序列长度（默认20）
- batch_size: 批次大小（默认32）
- max_epochs: 最大训练轮数（默认30）
- learning_rate: 学习率（默认0.001）
- hidden_dim: 隐藏层维度
- num_layers: Transformer层数
- num_heads: 注意力头数

### 预测和可视化

训练完成后，可以使用模型进行预测：

```bash
python predict.py
```

这将：
- 加载训练好的模型
- 使用测试数据进行预测
- 为每个特征生成预测结果的可视化图表
- 保存预测结果图表到当前目录

预测结果图表包含：
- 真实序列（蓝色实线）
- 预测序列（红色虚线）
- 预测起始点标记（绿色竖线）

## 模型结构

- **输入投影**：将输入特征映射到模型的隐藏维度
- **位置编码**：添加位置信息
- **Transformer层**：多个Transformer编码器层，每层包含：
  - 多头自注意力机制（带有因果掩码）
  - 层归一化
  - 前馈神经网络
- **输出投影**：将隐藏表示映射回输出特征维度

## 测试

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


