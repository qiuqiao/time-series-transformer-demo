# 使用多层因果Transformer与高斯混合模型进行时间序列预测

## 简介

非常简单的小项目，仅仅用于完成学校的任务

这是一个基于PyTorch Lightning实现的用于自回归时间序列预测的Transformer模型，结合了高斯混合模型（GMM）来建模预测的不确定性。输入为(B,T,C)，B为批次大小，T为序列长度，C为特征数量。模型对每个特征和时间步都输出多个高斯分布的参数（均值和方差），通过采样和平均得到最终预测值。


## 特点

- 多层Transformer架构
- 因果注意力机制，确保每个时间步只能看到当前及之前的信息
- 高斯混合模型支持：
  - 可配置的高斯核数量
  - 为每个预测输出多个高斯分布
  - 通过重参数化技巧实现可导的采样过程
  - 同时优化重构损失和负对数似然损失
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

### CSV数据支持

模型支持使用CSV数据文件进行训练。CSV中的每一行代表一个时间步，每一列代表一个特征。特征列应命名为"feature_1"、"feature_2"等。

1. 生成一个玩具数据集：

```bash
python generate_csv_data.py
```

这将在`data`目录中创建训练集、验证集和测试集。

2. 使用CSV数据训练模型：

多个csv：在`data/train_data`目录中准备好多个csv文件，然后运行：

```bash
python preprocess.py
```

会自动处理nan值，并把处理后的文件保存到`data/train_data_processed`目录中

单个csv：在`data`目录中准备好`train_data.csv`

无论是单个还是多个csv，需要同时在`data`目录中准备好`val_data.csv`

然后运行：

```bash
python train.py
```

训练参数可在train.py中修改：
- seq_len: 输入序列长度（默认500）
- pred_len: 预测序列长度（默认500）
- batch_size: 批次大小（默认32）
- max_epochs: 最大训练轮数（默认30）
- learning_rate: 学习率（默认0.0008）
- hidden_dim: 隐藏层维度（默认256）
- num_layers: Transformer层数（默认10）
- num_heads: 注意力头数（默认8）
- num_gmm_kernels: 高斯核数量（默认5）

可以使用tensorboard可视化训练过程：

```bash
tensorboard --logdir=lightning_logs
```

### 损失函数

模型使用两个损失函数的组合：
1. 重构损失（MSE）：衡量预测值与真实值的距离
2. 负对数似然损失：衡量预测分布对真实值的解释能力

总损失 = 重构损失 + 负对数似然损失

这种组合允许模型不仅学习准确的预测值，还能估计预测的不确定性。

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
- `--seq_len`：输入序列长度（默认：500）
- `--pred_len`：预测序列长度（默认：500）
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
    --seq_len 500 \
    --pred_len 500 \
    --output_dir my_predictions
```

预测过程说明：
1. 模型采用自回归（autoregressive）方式生成预测序列
2. 每个预测步骤都会生成多个高斯分布的参数
3. 在高斯混合分布中随机采样，作为最终预测值
4. 输入序列会被限制在 `seq_len` 长度内（过长则裁剪），以确保与训练时的输入长度匹配
5. 每次预测时会使用之前的预测结果作为下一步预测的输入

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
- **输出投影**：将隐藏表示映射到高斯混合模型参数空间
  - 输出维度为 num_features * num_gmm_kernels * 2
  - 为每个特征预测多个高斯分布的均值和方差
- **采样层**：使用重参数化技巧从预测的分布中采样
- **聚合层**：将多个高斯核的采样结果平均得到最终预测

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


