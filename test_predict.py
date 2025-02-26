import os
import pytest
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule

# 导入要测试的模块
from predict import (
    load_model,
    predict_sequence,
    plot_prediction_results,
    save_prediction_results,
)


# 创建一个模拟的模型类，用于测试
class MockTransformerModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def predict_sequence(self, initial_seq, pred_len):
        # 模拟预测，返回一个与initial_seq形状相似但长度增加pred_len的序列
        # 处理不同形状的输入
        if isinstance(initial_seq, torch.Tensor):
            # 如果是张量，转换为numpy数组
            if len(initial_seq.shape) == 3:  # (batch_size, seq_len, feature_dim)
                initial_np = initial_seq[0].numpy()  # 取第一个批次
            else:  # (seq_len, feature_dim)
                initial_np = initial_seq.numpy()
        else:
            # 如果已经是numpy数组
            initial_np = initial_seq

        # 获取序列长度和特征维度
        seq_len = initial_np.shape[0]
        feature_dim = initial_np.shape[1]

        # 创建一个简单的预测序列
        pred_seq = np.zeros((seq_len + pred_len, feature_dim))
        pred_seq[:seq_len] = initial_np

        # 后面的预测部分使用简单的线性增长
        for i in range(pred_len):
            idx = seq_len + i
            pred_seq[idx] = pred_seq[seq_len - 1] + (i + 1) * 0.1

        return pred_seq


@pytest.fixture
def mock_model():
    return MockTransformerModel()


@pytest.fixture
def sample_data():
    # 创建一个简单的测试数据集
    seq_len = 50
    feature_dim = 3

    # 生成时间步
    t = np.linspace(0, 10, seq_len)

    # 创建DataFrame
    df = pd.DataFrame()
    df["time_step"] = t

    # 为每个特征生成不同的信号
    for i in range(feature_dim):
        # 简单的正弦波
        signal = np.sin(t * (i + 1))
        df[f"feature_{i+1}"] = signal

    return df


def test_load_model(monkeypatch):
    """测试模型加载功能"""

    # 模拟TransformerLightningModule类
    class MockTransformerLightningModule:
        def __init__(self, **kwargs):
            self.hparams = kwargs
            self.eval_called = False

        def load_state_dict(self, state_dict, strict=True):
            # 简单地接受任何状态字典
            return None

        def eval(self):
            self.eval_called = True
            return self

    # 模拟torch.load函数
    def mock_torch_load(path):
        assert path.endswith(".ckpt"), "应该加载.ckpt文件"
        return {
            "state_dict": {"model.key": "value"},
            "hyper_parameters": {
                "input_dim": 3,
                "hidden_dim": 64,
                "output_dim": 3,
                "num_layers": 3,
                "num_heads": 4,
            },
        }

    # 应用模拟
    monkeypatch.setattr(torch, "load", mock_torch_load)
    monkeypatch.setattr(
        "predict.TransformerLightningModule", MockTransformerLightningModule
    )

    # 测试加载模型
    model = load_model("checkpoints/test_model.ckpt")

    # 验证返回的是正确的类型
    assert isinstance(
        model, MockTransformerLightningModule
    ), "load_model应该返回正确的模型类型"

    # 验证模型已设置为评估模式
    assert model.eval_called, "模型应该被设置为评估模式"


def test_predict_sequence(mock_model, sample_data):
    """测试序列预测功能"""
    # 导入要测试的函数
    from predict import predict_sequence

    # 准备输入数据
    initial_seq = sample_data.iloc[:30][["feature_1", "feature_2", "feature_3"]].values
    pred_len = 20

    # 调用预测函数
    result = predict_sequence(mock_model, initial_seq, pred_len)

    # 验证结果
    assert isinstance(result, tuple), "predict_sequence应该返回一个元组"
    assert len(result) == 2, "predict_sequence应该返回两个元素"

    true_seq, pred_seq = result

    # 验证真实序列和预测序列的形状
    assert true_seq.shape[1] == initial_seq.shape[1], "真实序列的特征维度应该与输入相同"
    assert pred_seq.shape[1] == initial_seq.shape[1], "预测序列的特征维度应该与输入相同"
    assert pred_seq.shape[0] >= initial_seq.shape[0], "预测序列的长度应该不小于输入序列"


def test_plot_prediction_results(mock_model, sample_data, monkeypatch):
    """测试绘图功能"""
    # 模拟plt函数
    figure_called = False
    savefig_called = False

    class MockFigure:
        def __init__(self, *args, **kwargs):
            pass

    def mock_figure(*args, **kwargs):
        nonlocal figure_called
        figure_called = True
        return MockFigure()

    def mock_plot(*args, **kwargs):
        return []

    def mock_axvline(*args, **kwargs):
        return []

    def mock_title(*args, **kwargs):
        pass

    def mock_xlabel(*args, **kwargs):
        pass

    def mock_ylabel(*args, **kwargs):
        pass

    def mock_legend(*args, **kwargs):
        pass

    def mock_grid(*args, **kwargs):
        pass

    def mock_savefig(filename, *args, **kwargs):
        nonlocal savefig_called
        savefig_called = True
        assert filename.endswith(".png"), "保存的文件应该是PNG格式"

    def mock_close(*args, **kwargs):
        pass

    # 应用模拟
    monkeypatch.setattr(plt, "figure", mock_figure)
    monkeypatch.setattr(plt, "plot", mock_plot)
    monkeypatch.setattr(plt, "axvline", mock_axvline)
    monkeypatch.setattr(plt, "title", mock_title)
    monkeypatch.setattr(plt, "xlabel", mock_xlabel)
    monkeypatch.setattr(plt, "ylabel", mock_ylabel)
    monkeypatch.setattr(plt, "legend", mock_legend)
    monkeypatch.setattr(plt, "grid", mock_grid)
    monkeypatch.setattr(plt, "savefig", mock_savefig)
    monkeypatch.setattr(plt, "close", mock_close)

    # 准备输入数据
    true_seq = sample_data[["feature_1", "feature_2", "feature_3"]].values
    pred_seq = np.copy(true_seq)  # 简单复制作为预测序列

    # 调用绘图函数
    plot_prediction_results(
        true_seq, pred_seq, feature_idx=0, pred_len=20, save_path="test_plot.png"
    )

    # 验证函数调用
    assert figure_called, "plt.figure应该被调用"
    assert savefig_called, "plt.savefig应该被调用"


def test_save_prediction_results(sample_data, tmp_path):
    """测试保存预测结果功能"""
    # 导入要测试的函数
    from predict import save_prediction_results

    # 准备输入数据
    true_seq = sample_data[["feature_1", "feature_2", "feature_3"]].values
    # 创建一个与true_seq长度相同的pred_seq
    pred_seq = np.copy(true_seq)  # 简单复制作为预测序列

    # 创建临时文件路径
    save_path = os.path.join(tmp_path, "prediction_results.csv")

    # 调用保存函数
    save_prediction_results(true_seq, pred_seq, save_path)

    # 验证文件是否已创建
    assert os.path.exists(save_path), "预测结果文件应该被创建"

    # 读取保存的文件并验证内容
    saved_data = pd.read_csv(save_path)

    # 验证列名
    expected_columns = ["time_step"]
    for i in range(true_seq.shape[1]):
        expected_columns.extend([f"true_feature_{i+1}", f"pred_feature_{i+1}"])

    assert (
        list(saved_data.columns) == expected_columns
    ), "保存的CSV文件应该包含正确的列名"

    # 验证行数
    assert len(saved_data) == len(true_seq), "保存的CSV文件应该包含正确的行数"


def test_main_function(monkeypatch, mock_model, sample_data):
    """测试主函数功能"""
    # 模拟函数调用
    load_model_called = False
    predict_sequence_called = False
    plot_prediction_results_called = False
    save_prediction_results_called = False

    def mock_load_model(model_path):
        nonlocal load_model_called
        load_model_called = True
        assert model_path.endswith(".ckpt"), "应该加载.ckpt文件"
        return mock_model

    def mock_load_test_data(data_path, seq_len):
        # 返回相同长度的初始序列和完整序列，避免长度不匹配问题
        initial_seq = sample_data[["feature_1", "feature_2", "feature_3"]].values
        full_seq = initial_seq.copy()
        return initial_seq, full_seq

    def mock_predict_sequence(model, initial_seq, pred_len):
        nonlocal predict_sequence_called
        predict_sequence_called = True
        assert model == mock_model, "应该使用加载的模型"
        assert pred_len > 0, "预测长度应该大于0"
        # 返回相同长度的序列
        return initial_seq, initial_seq

    def mock_plot_prediction_results(
        true_seq, pred_seq, feature_idx, pred_len, save_path
    ):
        nonlocal plot_prediction_results_called
        plot_prediction_results_called = True
        assert true_seq is not None, "真实序列不应为None"
        assert pred_seq is not None, "预测序列不应为None"
        assert save_path.endswith(".png"), "保存路径应该是PNG格式"

    def mock_save_prediction_results(true_seq, pred_seq, save_path):
        nonlocal save_prediction_results_called
        save_prediction_results_called = True
        assert true_seq is not None, "真实序列不应为None"
        assert pred_seq is not None, "预测序列不应为None"
        assert save_path.endswith(".csv"), "保存路径应该是CSV格式"

    # 模拟命令行参数
    class MockArgs:
        def __init__(self):
            self.model_path = "checkpoints/best_model.ckpt"
            self.data_path = "data/test_data.csv"
            self.seq_len = 50
            self.pred_len = 20
            self.output_dir = "predictions"

    def mock_parse_args():
        return MockArgs()

    # 应用模拟
    monkeypatch.setattr("predict.load_model", mock_load_model)
    monkeypatch.setattr("predict.load_test_data", mock_load_test_data)
    monkeypatch.setattr("predict.predict_sequence", mock_predict_sequence)
    monkeypatch.setattr("predict.plot_prediction_results", mock_plot_prediction_results)
    monkeypatch.setattr("predict.save_prediction_results", mock_save_prediction_results)
    monkeypatch.setattr("predict.parse_args", mock_parse_args)
    monkeypatch.setattr("os.makedirs", lambda path, exist_ok: None)

    # 导入并调用主函数
    from predict import main

    main()

    # 验证函数调用
    assert load_model_called, "load_model应该被调用"
    assert predict_sequence_called, "predict_sequence应该被调用"
    assert plot_prediction_results_called, "plot_prediction_results应该被调用"
    assert save_prediction_results_called, "save_prediction_results应该被调用"
