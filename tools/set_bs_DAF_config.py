import argparse
import re
import os

def parse_command_line_args():
    """解析命令行参数：-p 指定文件路径，--config 指定配置键值对"""
    parser = argparse.ArgumentParser(description="修改mmdet3d配置文件指定参数")
    # -p 参数：配置文件路径（必填）
    parser.add_argument("-p", "--path", required=True, type=str, help="配置文件的绝对/相对路径")
    # --config 参数：要修改的配置项（键值对，格式如 samples_per_gpu=16 num_epochs=10 use_deformable_func=False）
    parser.add_argument("--config", required=True, nargs="+", help="要修改的配置项，格式为 key=value，多个用空格分隔")
    
    args = parser.parse_args()
    return args

def parse_config_dict(config_list):
    """将--config传入的列表解析为dict，并进行类型转换"""
    config_dict = {}
    allowed_keys = {"samples_per_gpu", "num_epochs", "use_deformable_func"}  # 支持的配置项
    
    for item in config_list:
        if "=" not in item:
            print(f"⚠️ Warning: 配置项 {item} 格式错误，需遵循 key=value 格式，已跳过")
            continue
        
        key, value_str = item.split("=", 1)  # 1表示只分割第一个=，避免value中包含=
        key = key.strip()
        value_str = value_str.strip()
        
        # 检查是否是支持的配置项
        if key not in allowed_keys:
            print(f"⚠️ Warning: 配置项 {key} 不在支持的列表中（支持：{allowed_keys}），已跳过")
            continue
        
        # 类型转换（对应配置项的正确类型）
        try:
            if key in ["samples_per_gpu", "num_epochs"]:
                # 转换为整数
                value = int(value_str)
            elif key == "use_deformable_func":
                # 转换为布尔值（严格匹配 True/False 字符串）
                if value_str in ["True", "False"]:
                    value = eval(value_str)  # 安全转换Python布尔值
                else:
                    raise ValueError("布尔值只能是 True 或 False")
            else:
                value = value_str  # 备用（无实际意义，已限制allowed_keys）
            
            config_dict[key] = value
        except ValueError as e:
            print(f"⚠️ Warning: 配置项 {key} 的值 {value_str} 类型错误，错误信息：{e}，已跳过")
    
    return config_dict

def modify_config_file(file_path, config_dict):
    """
    核心功能：修改配置文件指定项（仅匹配独立的 key = 赋值行，取第一个匹配，保留其余格式）
    Args:
        file_path: 配置文件路径
        config_dict: 要修改的键值对dict
    """
    # 验证文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ Error: 文件 {file_path} 不存在，无法修改")
        return
    
    # 允许的配置项（用于后续匹配，避免误改）
    allowed_keys = {"samples_per_gpu", "num_epochs", "use_deformable_func"}
    # 记录已找到并修改的配置项（确保每个key只修改第一个匹配项）
    found_and_modified = set()
    
    # 1. 读取文件所有行（保留原格式）
    with open(file_path, "r", encoding="utf-8") as f:
        file_lines = f.readlines()
    
    # 2. 逐行处理，修改目标配置项（仅匹配独立的 key = 赋值行）
    modified_lines = []
    for line in file_lines:
        modified_line = line  # 默认保留原行
        
        # 遍历要修改的配置项，匹配并替换（仅处理未修改过的key）
        for key, new_value in config_dict.items():
            # 跳过：非允许项、已修改过该key、该行已被其他key修改
            if key not in allowed_keys or key in found_and_modified or modified_line != line:
                continue
            
            # 核心优化：正则严格匹配「独立的 key = 赋值行」
            # 匹配规则：
            # 1. ^\s*：行首可选缩进
            # 2. ({key})：精确匹配key（完整单词，不被其他字符包裹）
            # 3. \s+=\s+：等号两边至少有一个空格（区别于 key=value 或 key*=value）
            # 4. .*$：后续的赋值内容（常量，非表达式）
            # 该正则不会匹配：变量引用、表达式、无空格赋值行
            pattern = re.compile(rf"^(\s*)({key})(\s+=\s+)(.*)$")
            match = pattern.match(line)
            
            if match:
                # 捕获分组：1=缩进，2=key，3=「 = 」（含两边空格），4=原有值
                indent = match.group(1)
                key_str = match.group(2)
                assign_str = match.group(3)  # 即「 = 」（保留原空格格式）
                
                # 转换新值为合适的字符串格式（保持Python语法）
                if isinstance(new_value, bool):
                    new_value_str = str(new_value)  # 布尔值直接转成 "True"/"False"
                else:
                    new_value_str = str(new_value)  # 整数转字符串
                
                # 构造新行（保留原缩进、等号两边空格格式）
                modified_line = f"{indent}{key_str}{assign_str}{new_value_str}\n"
                
                # 标记该key已找到并修改（后续不再匹配该key的其他行）
                found_and_modified.add(key)
                break  # 找到对应key，无需继续匹配其他key
        
        modified_lines.append(modified_line)
    
    # 3. 打印未找到的配置项警告
    for key in config_dict:
        if key not in found_and_modified:
            print(f"⚠️ Warning: 配置项 {key} 未找到独立的「{key} = 」赋值行，未进行修改")
    
    # 4. 将修改后的内容写回文件（覆盖原文件，如需备份可先复制文件）
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(modified_lines)
    
    print(f"\n✅ 操作完成！文件 {file_path} 已修改（修改项：{found_and_modified}）")

def main():
    # 步骤1：解析命令行参数
    args = parse_command_line_args()
    
    # 步骤2：解析配置dict
    config_dict = parse_config_dict(args.config)
    if not config_dict:
        print("ℹ️ Info: 无有效配置项需要修改，程序退出")
        return
    
    # 步骤3：修改配置文件
    modify_config_file(args.path, config_dict)

if __name__ == "__main__":
    main()