import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
np.random.seed(42)


def generate_time_series(seq_len, num_features=3):
    """
    生成多特征时间序列数据

    参数:
        seq_len: 序列长度（时间步数量）
        num_features: 特征数量

    返回:
        DataFrame，每行是一个时间步，每列是一个特征
    """
    # 生成时间步
    t = np.linspace(0, 50, seq_len)

    # 创建DataFrame
    df = pd.DataFrame()
    df["time_step"] = t

    # 为每个特征生成不同的信号
    for i in range(num_features):
        # 生成线性趋势
        slope = np.random.uniform(-0.2, 0.2)
        intercept = np.random.uniform(-1, 1)
        linear = slope * t + intercept

        # 生成正弦波 (每个特征使用不同的参数)
        amplitude = np.random.uniform(0.5, 1.5)
        frequency = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        sine = amplitude * np.sin(frequency * t + phase)

        # 添加噪声
        noise = np.random.normal(0, 0.1, len(t))

        # 组合信号
        signal = linear + sine + noise

        # 添加到DataFrame
        df[f"feature_{i+1}"] = signal

    return df


def save_dataset(data, filename):
    """保存数据集到CSV文件"""
    data.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}, total {len(data)} rows")


def plot_samples(data, filename="sample_data.png"):
    """绘制数据样本"""
    plt.figure(figsize=(12, 8))

    # 获取特征列（排除time_step列）
    feature_cols = [col for col in data.columns if col.startswith("feature")]

    # 为每个特征绘制一个子图
    for i, feature in enumerate(feature_cols):
        plt.subplot(len(feature_cols), 1, i + 1)
        plt.plot(data["time_step"], data[feature])
        plt.title(f"Feature {feature}")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Sample chart saved to {filename}")


def main():
    # 创建数据目录
    os.makedirs("data", exist_ok=True)

    # 参数设置
    seq_len_train = 1000  # 时间步数量
    seq_len_val = 120  # 时间步数量
    seq_len_test = 120  # 时间步数量
    num_features = 3  # 特征数量

    # 生成训练集
    print("Generating...")
    train_data = generate_time_series(seq_len=seq_len_train, num_features=num_features)
    random_index = np.random.randint(0, len(train_data) - seq_len_val - seq_len_test)
    val_data = train_data.iloc[random_index : random_index + seq_len_val]
    test_data = train_data.iloc[
        random_index + seq_len_val : random_index + seq_len_val + seq_len_test
    ]
    # train_data除了time_step这一个column，其他的值都加上随机噪声
    train_data.loc[:, train_data.columns != "time_step"] += np.random.normal(
        0, 0.1, train_data.shape
    )[:, train_data.columns != "time_step"]
    save_dataset(train_data, "data/train_data.csv")
    save_dataset(val_data, "data/val_data.csv")
    save_dataset(test_data, "data/test_data.csv")

    # 绘制样本
    print("Plotting samples...")
    plot_samples(train_data, filename="data/train_data.png")
    plot_samples(val_data, filename="data/val_data.png")
    plot_samples(test_data, filename="data/test_data.png")

    print("Data generation completed!")


if __name__ == "__main__":
    main()
