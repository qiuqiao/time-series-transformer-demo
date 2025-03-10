import os
import pandas as pd
import numpy as np
import glob
import argparse
import time
from tqdm import tqdm


def process_csv_file(file_path, output_dir):
    """处理单个CSV文件，删除含有NaN的行并拆分为多个文件"""
    file_name = os.path.basename(file_path)
    print(f"处理文件: {file_name}")

    try:
        # 获取文件大小，用于显示进度
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # 如果文件大于100MB
            print(f"  文件较大 ({file_size / (1024 * 1024):.1f} MB)，使用分块读取")
            # 使用分块读取大文件
            chunk_size = 10000  # 每次读取的行数
            reader = pd.read_csv(file_path, chunksize=chunk_size)

            # 初始化变量
            all_chunks = []
            current_chunk = []
            chunk_num = 1
            total_rows_processed = 0

            for chunk in tqdm(reader, desc="  读取数据块"):
                for i, row in chunk.iterrows():
                    total_rows_processed += 1

                    # 如果行中有NaN值，则保存当前块并开始新块
                    if row.isna().any():
                        if current_chunk:
                            # 保存当前块
                            chunk_df = pd.DataFrame(current_chunk)
                            base_name = os.path.splitext(file_name)[0]
                            output_file = os.path.join(
                                output_dir, f"{base_name}_chunk{chunk_num}.csv"
                            )
                            chunk_df.to_csv(output_file, index=False)
                            print(
                                f"  保存块 {chunk_num}: 包含 {len(current_chunk)} 行到文件 {os.path.basename(output_file)}"
                            )

                            # 更新块编号并清空当前块
                            chunk_num += 1
                            current_chunk = []

                        # print(f"  跳过行 {total_rows_processed}，因为包含NaN值")
                    else:
                        # 添加到当前块
                        current_chunk.append(row.to_dict())

            # 保存最后一个块（如果有）
            if current_chunk:
                chunk_df = pd.DataFrame(current_chunk)
                base_name = os.path.splitext(file_name)[0]
                output_file = os.path.join(
                    output_dir, f"{base_name}_chunk{chunk_num}.csv"
                )
                chunk_df.to_csv(output_file, index=False)
                print(
                    f"  保存块 {chunk_num}: 包含 {len(current_chunk)} 行到文件 {os.path.basename(output_file)}"
                )

        else:
            # 对于小文件，一次性读取
            df = pd.read_csv(file_path)
            print(f"  文件大小: {file_size / 1024:.1f} KB，共 {len(df)} 行")

            # 初始化变量
            start_idx = 0
            chunk_num = 1

            # 遍历数据行
            for i in tqdm(range(len(df) + 1), desc="  处理行"):
                # 当到达文件末尾或遇到含有NaN的行时
                if i == len(df) or df.iloc[i].isna().any():
                    # 如果当前块有有效行，则保存
                    if i > start_idx:
                        chunk_df = df.iloc[start_idx:i]

                        # 创建输出文件名
                        base_name = os.path.splitext(file_name)[0]
                        output_file = os.path.join(
                            output_dir, f"{base_name}_chunk{chunk_num}.csv"
                        )

                        # 保存到文件
                        chunk_df.to_csv(output_file, index=False)
                        print(
                            f"  保存块 {chunk_num}: 行 {start_idx+1} 到 {i} 到文件 {os.path.basename(output_file)}"
                        )

                        # 更新块编号
                        chunk_num += 1

                    # 如果当前行有NaN，跳过该行
                    if i < len(df) and df.iloc[i].isna().any():
                        # print(f"  跳过行 {i+1}，因为包含NaN值")
                        start_idx = i + 1

        return True
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {str(e)}")
        return False


def process_csv_files(test_mode=False):
    """处理所有CSV文件"""
    start_time = time.time()

    # 定义输入和输出目录
    if test_mode:
        input_files = ["test_sample.csv"]
        output_dir = "test_output"
    else:
        input_dir = "data/train_data"
        output_dir = "data/train_data_processed"
        input_files = glob.glob(os.path.join(input_dir, "*.csv"))

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"找到 {len(input_files)} 个CSV文件需要处理")

    # 处理所有文件
    success_count = 0
    for file_path in input_files:
        if process_csv_file(file_path, output_dir):
            success_count += 1

    # 打印总结
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n处理完成！")
    print(f"总共处理了 {len(input_files)} 个文件，成功 {success_count} 个")
    print(f"总耗时: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="处理CSV文件，删除含有NaN的行并拆分为多个文件"
    )
    parser.add_argument(
        "--test", action="store_true", help="运行测试模式，处理test_sample.csv文件"
    )
    args = parser.parse_args()

    process_csv_files(test_mode=args.test)
