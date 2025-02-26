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

    def __init__(self, csv_file, seq_len, pred_len=1):
        """
        初始化数据集

        参数:
            csv_file: CSV文件路径
            seq_len: 输入序列长度
            pred_len: 预测序列长度
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data = self._load_data(csv_file)
        print(f"Loaded {len(self.data)} samples from {csv_file}")

    def _load_data(self, csv_file):
        """加载CSV数据并处理成序列"""
        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 获取特征列（排除time_step列）
        feature_cols = [col for col in df.columns if col.startswith("feature")]

        # 提取特征数据
        features = df[feature_cols].values  # (seq_len, num_features)

        # 创建滑动窗口样本
        data = []

        # 检查数据长度是否足够
        if len(df) <= self.seq_len + self.pred_len:
            print(
                f"警告: {csv_file} 的数据长度 ({len(df)}) 小于序列长度+预测长度 ({self.seq_len + self.pred_len})，无法创建样本"
            )
            return data

        # 创建滑动窗口样本，每个样本包含输入序列和目标序列
        for i in range(len(df) - self.seq_len - self.pred_len + 1):
            # 输入序列: (seq_len, num_features)
            input_seq = features[i : i + self.seq_len]
            # 目标序列: (seq_len, num_features) - 包括输入序列和预测序列
            # 这样模型就会学习预测整个序列，而不仅仅是下一个时间步
            target_seq = features[i + 1 : i + self.seq_len + 1]

            data.append((input_seq, target_seq))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class TransformerLightningModule(pl.LightningModule):
    """PyTorch Lightning 模块，封装 CausalTransformer"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        num_heads,
        learning_rate=0.001,
        ff_dim=None,
        dropout=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 创建模型
        self.model = CausalTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # 损失函数
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
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
                pred = self(input_seq)
                next_val = pred[:, -1, :].cpu().numpy()  # 获取最后一个预测值并转为numpy

                # 将预测值添加到输出序列
                output_seq = np.vstack([output_seq, next_val[0]])

                # 更新输入序列（滑动窗口）
                if len(initial_seq) + i + 1 <= len(initial_seq):
                    # 如果还在初始序列范围内，使用真实值
                    input_seq = torch.FloatTensor(
                        output_seq[-len(initial_seq) :]
                    ).unsqueeze(0)
                else:
                    # 否则使用包含预测值的序列
                    input_seq = torch.FloatTensor(
                        output_seq[-len(initial_seq) :]
                    ).unsqueeze(0)

        return output_seq


def load_test_sequence(csv_file, seq_len, pred_len):
    """从CSV文件加载测试序列"""
    df = pd.read_csv(csv_file)

    # 获取特征列（排除time_step列）
    feature_cols = [col for col in df.columns if col.startswith("feature")]

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
    seq_len = 100  # 恢复原来的输入序列长度
    pred_len = 20  # 预测序列长度
    batch_size = 32
    max_epochs = 30

    # 检查数据目录是否存在
    if not os.path.exists("data"):
        print("Data directory does not exist, generating data...")
        import generate_csv_data

        generate_csv_data.main()

    # 创建数据集
    print("Loading datasets...")
    train_dataset = CSVDataset(csv_file="data/train_data.csv", seq_len=seq_len)
    val_dataset = CSVDataset(csv_file="data/val_data.csv", seq_len=seq_len)
    test_dataset = CSVDataset(csv_file="data/test_data.csv", seq_len=seq_len)

    # 获取特征维度
    sample_x, _ = train_dataset[0]
    input_dim = sample_x.shape[1]  # 特征数量
    output_dim = input_dim  # 输出维度与输入维度相同

    # 模型参数
    hidden_dim = 32
    num_layers = 5
    num_heads = 4

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # 创建模型
    print("Initializing model...")
    model = TransformerLightningModule(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        learning_rate=0.001,
        dropout=0.1,
    )

    # 设置回调
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="transformer-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    # 设置日志记录器
    logger = TensorBoardLogger("lightning_logs", name="transformer")

    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        accelerator="auto",
        val_check_interval=1.0,  # 每个epoch验证一次
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
