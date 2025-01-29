import os
import shutil
import re

def extract_prefix(line):
    """
    从给定的行中提取前缀（假设为 ID）。
    
    :param line: 需要解析的一行文本
    :return: 提取的前缀字符串
    """
    # 使用正则表达式匹配前缀部分
    match = re.match(r'^(\d+_\d+_\d+_\d+)', line)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Unable to extract prefix from line: {line}")

def build_audio_filepath(prefix, base_dir):
    """
    根据前缀构建音频文件的完整路径。
    
    :param prefix: 前缀字符串
    :param base_dir: 包含音频文件的基本目录
    :return: 音频文件的完整路径
    """
    parts = prefix.split('_')
    subdir = os.path.join(base_dir, parts[0], parts[1])
    filename = f"{prefix}.wav"
    return os.path.join(subdir, filename)

def copy_audio_files(text_file_path, audio_base_dir, output_dir):
    """
    根据文本文件中的前缀信息复制音频文件到目标文件夹。
    
    :param text_file_path: 文本文件路径
    :param audio_base_dir: 包含音频文件的基本目录
    :param output_dir: 目标文件夹路径
    """
    with open(text_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        try:
            prefix = extract_prefix(line.strip())
            src_file = build_audio_filepath(prefix, audio_base_dir)
            
            if not os.path.exists(src_file):
                print(f"Warning: Audio file does not exist at path {src_file}")
                continue
            
            dst_file = os.path.join(output_dir, os.path.basename(src_file))
            shutil.copy2(src_file, dst_file)  # 使用 copy2 以保留元数据
            print(f"Copied {src_file} to {dst_file}")
        except Exception as e:
            print(f"Error processing line '{line.strip()}': {e}")

if __name__ == '__main__':
    text_file_path = 'filtered_sampled_output_2.txt'  # 替换为你的文本文件路径
    audio_base_dir = ''  # 替换为包含音频文件的基本目录
    output_dir = 'audio_2'  # 替换为目标文件夹路径
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    copy_audio_files(text_file_path, audio_base_dir, output_dir)