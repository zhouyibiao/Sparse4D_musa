import os
import re
import time
import argparse
from typing import List, Optional

def extract_time_from_line(line: str) -> Optional[float]:
    """
    从单行日志中提取时间数值
    :param line: 日志行字符串
    :return: 提取到的时间值（浮点数），无匹配则返回None
    """
    # 匹配 "cost time : 数字 s" 格式的正则表达式
    pattern = r'train per batch_data cost time : ([\d\.]+) s'
    match = re.search(pattern, line)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def process_log_file(file_path: str) -> None:
    """
    处理单个日志文件：提取时间、计算平均值、写入结果
    :param file_path: 日志文件路径
    """
    # 存储提取到的时间值
    time_values: List[float] = []
    
    # 读取文件并提取时间值
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 遍历每一行提取时间
        for line in lines:
            time_val = extract_time_from_line(line.strip())
            if time_val is not None:
                time_values.append(time_val)
        
        # 没有提取到足够的时间值（至少需要2个：去除第一个后还有数据）
        if len(time_values) <= 1:
            print(f"⚠️ 文件 {file_path}：提取到的时间值数量不足（共{len(time_values)}个），跳过计算")
            return
        
        # 去除第一个时间值，计算平均值
        filtered_times = time_values[1:]
        avg_time = sum(filtered_times) / len(filtered_times)
        
        # 准备写入的内容
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        result_lines = [
            "\n" + "="*50,
            f"计算时间：{current_time}",
            f"提取的所有时间值：{time_values}",
            f"去除第一个时间值后的数据：{filtered_times}",
            f"去除第一个时间值后的平均时间：{avg_time:.6f} s/epoch",
            "="*50 + "\n"
        ]
        
        # 将结果写入文件末尾
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(result_lines))
        
        print(f"✅ 文件 {file_path} 处理完成：")
        print(f"   - 提取时间值总数：{len(time_values)}")
        print(f"   - 去除第一个后数量：{len(filtered_times)}")
        print(f"   - 平均值：{avg_time:.6f} 秒")
        
    except Exception as e:
        print(f"❌ 处理文件 {file_path} 时出错：{str(e)}")

def batch_process_logs(root_path: str) -> None:
    """
    批量处理指定路径下的所有.log文件
    :param root_path: 根路径（可以是文件夹或单个.log文件）
    """
    # 检查路径是否存在
    if not os.path.exists(root_path):
        print(f"❌ 路径不存在：{root_path}")
        return
    
    # 如果是单个文件且是.log文件，直接处理
    if os.path.isfile(root_path) and root_path.endswith('.log'):
        process_log_file(root_path)
    # 如果是文件夹，遍历所有.log文件
    elif os.path.isdir(root_path):
        log_files = [
            os.path.join(root_path, filename)
            for filename in os.listdir(root_path)
            if filename.endswith('.log')
        ]
        
        if not log_files:
            print(f"⚠️ 路径 {root_path} 下未找到.log文件")
            return
        
        print(f"📁 找到 {len(log_files)} 个.log文件，开始处理...")
        for log_file in log_files:
            process_log_file(log_file)
            print("-" * 80)
    else:
        print(f"❌ 路径 {root_path} 不是有效的文件或文件夹")

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="日志文件时间提取工具：提取.log文件中train per batch_data耗时，去除第一个值后计算平均值并写入文件末尾",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 添加路径参数（支持短选项-p和长选项--path）
    parser.add_argument(
        '-p', '--path',
        required=True,
        type=str,
        help='目标路径：可以是.log文件路径 或 包含.log文件的文件夹路径\n'
             '示例：\n'
             '  Windows: python %(prog)s -p "C:\\logs\\train"\n'
             '  Linux/Mac: python %(prog)s --path "/home/user/logs"'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 开始处理
    print("🚀 开始处理日志文件...")
    batch_process_logs(args.path)
    print("🎉 所有文件处理完成！")

if __name__ == "__main__":
    main()