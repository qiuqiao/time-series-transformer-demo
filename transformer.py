import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头自注意力机制，支持因果掩码"""

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert (
            self.head_dim * num_heads == hidden_dim
        ), "hidden_dim必须能被num_heads整除"

        # 定义线性变换层
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # 添加一个小的epsilon值来防止数值不稳定
        self.eps = 1e-6

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 线性变换
        q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        k = self.key(x)  # (batch_size, seq_len, hidden_dim)
        v = self.value(x)  # (batch_size, seq_len, hidden_dim)

        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力分数，添加数值稳定性保护
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (scale + self.eps)

        # 将scores限制在一个合理的范围内
        scores = torch.clamp(scores, min=-100, max=100)

        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == float("-inf"), -1e4)

        # 应用softmax，添加数值稳定性
        attn_weights = F.softmax(scores, dim=-1)
        # 防止注意力权重为0
        attn_weights = attn_weights + self.eps
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        context = torch.matmul(
            attn_weights, v
        )  # (batch_size, num_heads, seq_len, head_dim)

        # 重塑回原始形状
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )  # (batch_size, seq_len, hidden_dim)

        # 最终线性变换
        output = self.output(context)  # (batch_size, seq_len, hidden_dim)

        # 检查并处理任何潜在的NaN值
        if torch.isnan(output).any():
            output = torch.where(torch.isnan(output), torch.zeros_like(output), output)

        return output


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""

    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = PositionwiseFeedForward(hidden_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output = self.attention(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈子层
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, hidden_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # 注册为缓冲区（不是模型参数）
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CausalTransformer(nn.Module):
    """因果Transformer模型，用于自回归时间序列预测"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        num_heads,
        ff_dim=None,
        dropout=0.1,
    ):
        super().__init__()

        # 如果未指定前馈层维度，则默认为隐藏层维度的4倍
        if ff_dim is None:
            ff_dim = 4 * hidden_dim

        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 位置编码
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout=dropout)

        # Transformer层
        self.layers = nn.ModuleList(
            [
                TransformerLayer(hidden_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_attention_mask(self, seq_len):
        """生成因果注意力掩码"""
        # 创建下三角矩阵（包括对角线）
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # 将True转换为-inf，False转换为0
        mask = torch.zeros_like(mask, dtype=torch.float)
        mask = mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), float("-inf")
        )
        # 添加批次和头部维度
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        return mask

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)

        返回:
            输出张量，形状为 (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.size()

        # 生成因果注意力掩码
        mask = self.get_attention_mask(seq_len).to(x.device)

        # 输入投影
        x = self.input_projection(x)

        # 添加位置编码
        x = self.positional_encoding(x)

        # 通过所有Transformer层，一次性处理整个序列
        for layer in self.layers:
            x = layer(x, mask)

        # 输出投影
        output = self.output_projection(x)  # (batch_size, seq_len, output_dim)

        return output
