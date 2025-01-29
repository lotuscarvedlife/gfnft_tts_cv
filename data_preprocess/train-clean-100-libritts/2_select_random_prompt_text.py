import random

def filter_and_sample_lines(input_file_path, output_file_path, min_length=5, max_length=100, sample_size=1000):
    # 存储符合条件的行
    valid_lines = []

    # 读取所有行并过滤
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            parts = line.strip().split(' ', 1)  # 按第一个空格分割
            if len(parts) != 2:
                continue  # 忽略格式不正确的行
            
            prefix, text = parts
            if min_length <= len(text) <= max_length:
                valid_lines.append((prefix, text))

    # 随机选择指定数量的样本
    sampled_lines = random.sample(valid_lines, min(sample_size, len(valid_lines)))

    # 将选中的样本写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for prefix, text in sampled_lines:
            outfile.write(f"{prefix} {text}\n")

if __name__ == "__main__":
    input_file = 'combined_output.txt'  # 输入文件路径
    output_file = 'filtered_sampled_output_2.txt'  # 输出文件路径
    min_text_length = 30  # 最小文本长度
    max_text_length = 40  # 最大文本长度
    sample_count = 1000  # 抽样数量

    filter_and_sample_lines(input_file, output_file, min_text_length, max_text_length, sample_count)
    print(f"Filtered and sampled lines have been written to {output_file}")