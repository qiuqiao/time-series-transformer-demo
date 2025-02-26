import torch
import torch.nn as nn
import pytest
import numpy as np
from transformer import CausalTransformer

def test_transformer_shape():
    """测试Transformer模型的输出形状是否正确"""
    batch_size = 32
    seq_len = 24
    input_dim = 10
    hidden_dim = 64
    num_layers = 3
    num_heads = 4
    
    # 创建输入数据
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 初始化模型
    model = CausalTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,  # 输出维度与输入相同
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # 前向传播
    y = model(x)
    
    # 检查输出形状
    assert y.shape == (batch_size, seq_len, input_dim), f"输出形状应为 {(batch_size, seq_len, input_dim)}，但得到 {y.shape}"

def test_causal_attention_mask():
    """测试因果注意力掩码是否正确实现"""
    batch_size = 8
    seq_len = 10
    input_dim = 5
    hidden_dim = 32
    num_layers = 2
    num_heads = 2
    
    # 创建输入数据
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 初始化模型
    model = CausalTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # 获取模型的注意力掩码
    # 注意：这里假设模型有一个方法可以获取注意力掩码，实际实现可能不同
    attn_mask = model.get_attention_mask(seq_len)
    
    # 检查掩码形状
    expected_mask_shape = (1, 1, seq_len, seq_len)
    assert attn_mask.shape == expected_mask_shape, f"掩码形状应为 {expected_mask_shape}，但得到 {attn_mask.shape}"
    
    # 检查掩码是否为上三角矩阵（包括对角线）
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:  # 当前位置及之前的位置
                assert attn_mask[0, 0, i, j] == 0, f"位置 ({i}, {j}) 应为 0（可见），但得到 {attn_mask[0, 0, i, j]}"
            else:  # 未来位置
                assert attn_mask[0, 0, i, j] == float('-inf'), f"位置 ({i}, {j}) 应为 -inf（不可见），但得到 {attn_mask[0, 0, i, j]}"

def test_autoregressive_property():
    """测试模型的自回归性质，确保每个时间步只使用当前及之前的信息"""
    batch_size = 4
    seq_len = 8
    input_dim = 3
    hidden_dim = 16
    num_layers = 2
    num_heads = 2
    
    # 创建模型
    model = CausalTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # 创建两个不同的输入序列，它们在前半部分相同，后半部分不同
    x1 = torch.randn(batch_size, seq_len, input_dim)
    x2 = x1.clone()
    
    # 修改后半部分
    mid_point = seq_len // 2
    x2[:, mid_point:, :] = torch.randn(batch_size, seq_len - mid_point, input_dim)
    
    # 前向传播
    y1 = model(x1)
    y2 = model(x2)
    
    # 检查前半部分的输出是否相同（因为输入相同且模型是自回归的）
    assert torch.allclose(y1[:, :mid_point, :], y2[:, :mid_point, :]), "自回归属性测试失败：输入序列前半部分相同时，输出的前半部分应该相同"

def test_gradient_flow():
    """测试梯度是否正确流动"""
    batch_size = 4
    seq_len = 6
    input_dim = 3
    hidden_dim = 16
    num_layers = 2
    num_heads = 2
    
    # 创建输入数据和目标
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    target = torch.randn(batch_size, seq_len, input_dim)
    
    # 初始化模型
    model = CausalTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # 前向传播
    y = model(x)
    
    # 计算损失
    loss = nn.MSELoss()(y, target)
    
    # 反向传播
    loss.backward()
    
    # 检查输入的梯度是否存在且不为零
    assert x.grad is not None, "输入的梯度不应为None"
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "输入的梯度不应全为零"
    
    # 检查模型参数的梯度
    for name, param in model.named_parameters():
        assert param.grad is not None, f"参数 {name} 的梯度不应为None"
        # 某些参数可能有零梯度是正常的，所以我们不检查是否全为零

def test_multi_step_prediction():
    """测试模型在多步预测中的表现"""
    batch_size = 2
    seq_len = 12
    input_dim = 4
    hidden_dim = 32
    num_layers = 2
    num_heads = 2
    
    # 创建模型
    model = CausalTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # 创建初始输入序列
    x_init = torch.randn(batch_size, 1, input_dim)
    
    # 自回归生成序列
    generated_seq = x_init
    for _ in range(seq_len - 1):
        # 使用当前序列预测下一个时间步
        with torch.no_grad():
            next_step = model(generated_seq)[:, -1:, :]
        # 将预测添加到序列中
        generated_seq = torch.cat([generated_seq, next_step], dim=1)
    
    # 检查生成序列的形状
    assert generated_seq.shape == (batch_size, seq_len, input_dim), f"生成序列形状应为 {(batch_size, seq_len, input_dim)}，但得到 {generated_seq.shape}"

def test_batch_independence():
    """测试批次之间的独立性"""
    batch_size = 4
    seq_len = 8
    input_dim = 3
    hidden_dim = 16
    num_layers = 2
    num_heads = 2
    
    # 创建模型
    model = CausalTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # 创建两个批次的输入
    x_batch = torch.randn(batch_size, seq_len, input_dim)
    
    # 分别处理每个样本
    individual_outputs = []
    for i in range(batch_size):
        with torch.no_grad():
            individual_output = model(x_batch[i:i+1])
        individual_outputs.append(individual_output)
    
    # 将单独处理的结果拼接起来
    individual_results = torch.cat(individual_outputs, dim=0)
    
    # 批量处理所有样本
    with torch.no_grad():
        batch_results = model(x_batch)
    
    # 检查结果是否相同
    assert torch.allclose(individual_results, batch_results, rtol=1e-5, atol=1e-5), "批处理结果应与单独处理每个样本的结果相同"

def test_model_components():
    """测试模型的各个组件是否正确初始化"""
    input_dim = 5
    hidden_dim = 32
    output_dim = 5
    num_layers = 2
    num_heads = 4
    
    model = CausalTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # 检查模型是否有正确数量的层
    assert hasattr(model, 'layers'), "模型应该有layers属性"
    assert len(model.layers) == num_layers, f"模型应有 {num_layers} 层，但有 {len(model.layers)} 层"
    
    # 检查输入和输出投影
    assert hasattr(model, 'input_projection'), "模型应该有input_projection属性"
    assert hasattr(model, 'output_projection'), "模型应该有output_projection属性"
    
    # 检查注意力头的数量
    for i, layer in enumerate(model.layers):
        assert hasattr(layer, 'attention'), f"第 {i+1} 层应该有attention属性"
        assert hasattr(layer.attention, 'num_heads'), f"第 {i+1} 层的attention应该有num_heads属性"
        assert layer.attention.num_heads == num_heads, f"第 {i+1} 层应有 {num_heads} 个注意力头，但有 {layer.attention.num_heads} 个" 