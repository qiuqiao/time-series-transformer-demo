import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pandas as pd
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from transformer import CausalTransformer

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)


class CSVDataset(Dataset):
    """从CSV文件加载时间序列数据集"""

    def __init__(self, csv_file, seq_len, pred_len=1, valid=False):
        """
        初始化数据集

        参数:
            csv_file: CSV文件路径
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            valid: 是否使用验证模式（True时使用分段方式而不是滑动窗口）
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.valid = valid
        self.data = self._load_data(csv_file)
        print(f"Loaded {len(self.data)} samples from {csv_file}")

    def _load_data(self, csv_file):
        """加载CSV数据并处理成序列"""
        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 获取特征列（除第一列外的所有列）
        feature_cols = df.columns[1:]

        # 提取特征数据
        features = df[feature_cols].values  # (seq_len, num_features)

        # 创建数据样本
        data = []

        # 检查数据长度是否足够
        if len(df) <= self.seq_len + self.pred_len:
            print(
                f"警告: {csv_file} 的数据长度 ({len(df)}) 小于序列长度+预测长度 ({self.seq_len + self.pred_len})，无法创建样本"
            )
            return data

        if self.valid:
            # 验证模式：将数据分成不重叠的段
            total_samples = (len(df) - self.pred_len) // self.seq_len
            for i in range(total_samples):
                start_idx = i * self.seq_len
                # 输入序列
                input_seq = features[start_idx : start_idx + self.seq_len]
                # 目标序列
                target_seq = features[start_idx + 1 : start_idx + self.seq_len + 1]
                data.append((input_seq, target_seq))
        else:
            # 训练模式：使用滑动窗口
            for i in range(len(df) - self.seq_len - self.pred_len + 1):
                # 输入序列: (seq_len, num_features)
                input_seq = features[i : i + self.seq_len]
                # 目标序列: (seq_len, num_features)
                target_seq = features[i + 1 : i + self.seq_len + 1]
                data.append((input_seq, target_seq))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class TransformerLightningModule(pl.LightningModule):
    """PyTorch Lightning 模块，封装 CausalTransformer 并添加高斯混合模型支持"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_features,
        num_layers,
        num_heads,
        num_gmm_kernels=5,
        learning_rate=0.001,
        ff_dim=None,
        dropout=0.1,
        warmup_steps=200,
        gamma=0.999,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_features = num_features
        self.num_gmm_kernels = num_gmm_kernels

        # 创建模型 - 输出维度为每个特征的每个高斯核的均值和方差
        self.model = CausalTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features
            * num_gmm_kernels
            * 2,  # 每个特征的每个高斯核都有均值和方差
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 获取transformer的输出
        output = self.model(
            x
        )  # (batch_size, seq_len, num_features * num_gmm_kernels * 2)

        batch_size, seq_len, _ = output.shape

        # 重塑输出以分离均值和方差
        output = output.reshape(
            batch_size, seq_len, self.num_features, self.num_gmm_kernels, 2
        )
        mu = output[..., 0]  # (batch_size, seq_len, num_features, num_gmm_kernels)
        logvar = output[..., 1]  # (batch_size, seq_len, num_features, num_gmm_kernels)

        res = self.reparameterize(
            mu, logvar
        )  # (batch_size, seq_len, num_features, num_gmm_kernels)
        res = res.mean(dim=-1)  # (batch_size, seq_len, num_features)

        return res, mu, logvar

    def gaussian_nll_loss(self, y_true, mu, logvar):
        """计算高斯负对数似然损失"""
        var = torch.exp(logvar)
        nll = 0.5 * (
            torch.log(2 * np.pi * var) + (y_true.unsqueeze(-1) - mu) ** 2 / var
        )
        return nll.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, mu, logvar = self(x)

        # 计算重构损失（MSE）
        recon_loss = F.mse_loss(y_hat, y)

        # 计算每个高斯核的负对数似然损失
        nll_loss = self.gaussian_nll_loss(y, mu, logvar)

        # 总损失
        loss = recon_loss + nll_loss

        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_recon_loss", recon_loss)
        self.log("train_nll_loss", nll_loss)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, mu, logvar = self(x)

        recon_loss = F.mse_loss(y_hat, y)
        nll_loss = self.gaussian_nll_loss(y, mu, logvar)
        loss = recon_loss + nll_loss

        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_recon_loss", recon_loss)
        self.log("val_nll_loss", nll_loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, mu, logvar = self(x)

        recon_loss = F.mse_loss(y_hat, y)
        nll_loss = self.gaussian_nll_loss(y, mu, logvar)
        loss = recon_loss + nll_loss

        self.log("test_loss", loss)
        self.log("test_recon_loss", recon_loss)
        self.log("test_nll_loss", nll_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

        # 创建学习率调度器
        def warmup_exponential_lr(step):
            if step < self.hparams.warmup_steps:
                # 线性预热
                return float(step) / float(max(1, self.hparams.warmup_steps))
            else:
                # 指数衰减
                return self.hparams.gamma ** ((step - self.hparams.warmup_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_exponential_lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 每个step更新学习率
                "frequency": 1,
            },
        }

    def predict_sequence(self, initial_seq, pred_len):
        """使用模型进行自回归预测"""
        self.eval()  # 确保模型处于评估模式

        with torch.no_grad():
            # 初始序列
            curr_seq = torch.FloatTensor(initial_seq).unsqueeze(
                0
            )  # (1, init_len, num_features)
            output_seq = initial_seq.copy()  # 使用copy避免修改原始数据

            # 当前输入序列
            input_seq = curr_seq.clone()

            # 逐步预测
            for i in range(pred_len):
                # 使用当前序列预测下一个值
                pred, _, _ = self(input_seq)
                next_val = pred[:, -1, :].cpu().numpy()

                # 将预测值添加到输出序列
                output_seq = np.vstack([output_seq, next_val[0]])

                # 更新输入序列（滑动窗口）
                input_seq = torch.FloatTensor(
                    output_seq[-len(initial_seq) :]
                ).unsqueeze(0)

        return output_seq


def load_test_sequence(csv_file, seq_len, pred_len):
    """从CSV文件加载测试序列"""
    df = pd.read_csv(csv_file)

    # 获取特征列（除第一列外的所有列）
    feature_cols = df.columns[1:]

    # 提取特征数据
    features = df[feature_cols].values  # (seq_len, num_features)

    # 获取初始序列和完整序列
    initial_seq = features[:seq_len]  # (seq_len, num_features)
    full_seq = features  # (total_len, num_features)

    return initial_seq, full_seq


def plot_results(
    true_seq, pred_seq, feature_idx=0, pred_len=20, title="Prediction Results"
):
    """绘制真实序列和预测序列的对比图"""
    plt.figure(figsize=(12, 6))

    # 选择要绘制的特征
    true_feature = true_seq[:, feature_idx]
    pred_feature = pred_seq[:, feature_idx]

    # 绘制真实值
    plt.plot(
        np.arange(len(true_feature)), true_feature, label="Ground Truth", color="blue"
    )

    # 绘制预测值（包括初始序列和预测部分）
    # 初始序列长度就是seq_len（在main函数中设置为100）
    initial_len = len(pred_feature) - pred_len  # 使用pred_len作为预测长度
    plt.plot(
        np.arange(len(pred_feature)),
        pred_feature,
        label="Prediction",
        color="red",
        linestyle="--",
    )

    # 标记分隔线
    plt.axvline(
        x=initial_len - 1, color="green", linestyle="-", label="Prediction Start"
    )

    plt.title(f"{title} - Feature {feature_idx+1}")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"prediction_result_feature_{feature_idx+1}.png")
    plt.show()


def main():
    # 参数设置
    seq_len = 500  # 恢复原来的输入序列长度
    pred_len = 500  # 预测序列长度
    batch_size = 32
    max_epochs = 30

    # 检查数据目录是否存在
    if not os.path.exists("data"):
        print("Data directory does not exist, generating data...")
        import generate_csv_data

        generate_csv_data.main()

    # 创建数据集
    print("Loading datasets...")
    train_dataset = CSVDataset(
        csv_file="data/train_data.csv", seq_len=seq_len, valid=False
    )
    val_dataset = CSVDataset(csv_file="data/val_data.csv", seq_len=seq_len, valid=True)
    test_dataset = CSVDataset(
        csv_file="data/test_data.csv", seq_len=seq_len, valid=True
    )

    # 获取特征维度
    sample_x, _ = train_dataset[0]
    input_dim = sample_x.shape[1]  # 特征数量
    output_dim = input_dim  # 输出维度与输入维度相同

    # 模型参数
    hidden_dim = 256
    num_layers = 10
    num_heads = 8

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # 创建模型
    print("Initializing model...")
    model = TransformerLightningModule(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_features=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        learning_rate=8e-4,
        dropout=0.1,
        warmup_steps=200,
        gamma=0.999,
    )

    # 设置回调
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="transformer-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )

    # 设置日志记录器
    logger = TensorBoardLogger("lightning_logs", name="transformer")

    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        accelerator="auto",
        precision="16-mixed",  # 启用自动混合精度训练
        val_check_interval=1.0,  # 每个epoch验证一次
        gradient_clip_val=1.0,  # 添加梯度裁剪阈值
        gradient_clip_algorithm="norm",  # 使用L2范数裁剪
    )

    # 训练模型
    print("Starting model training...")
    trainer.fit(model, train_loader, val_loader)

    # 测试模型
    print("Testing model...")
    trainer.test(model, test_loader)

    # 生成预测
    print("Generating predictions...")
    # 从测试集加载一个样本
    initial_seq, true_seq = load_test_sequence("data/test_data.csv", seq_len, pred_len)

    # 使用模型预测
    pred_seq = model.predict_sequence(initial_seq, pred_len)

    # 为每个特征绘制结果
    num_features = initial_seq.shape[1]
    for i in range(num_features):
        plot_results(
            true_seq,
            pred_seq,
            feature_idx=i,
            pred_len=pred_len,
            title="Transformer Time Series Prediction",
        )

    print(
        f"Completed! Prediction results saved as 'prediction_result_feature_X.png', where X is the feature index"
    )


if __name__ == "__main__":
    main()
