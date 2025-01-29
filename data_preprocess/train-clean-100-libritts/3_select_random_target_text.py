import random

def load_lines(file_path, remove_prefix=False):
    """加载文件中的所有行，可选地去掉前缀"""
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                prefix, text = parts
                if remove_prefix:
                    lines.append(text)
                else:
                    lines.append((prefix, text))
            elif not remove_prefix:
                lines.append(('', line.strip()))  # 如果没有前缀，则用空字符串代替
    return lines

def filter_by_length(lines, min_length=5, max_length=100):
    """根据长度过滤文本"""
    return [line for line in lines if min_length <= len(line[1]) <= max_length]

def main():
    combined_file = 'combined_output.txt'
    filtered_sampled_file = 'filtered_sampled_output_2.txt'
    # output_file = 'filtered_target_output_2.txt'
    output_file = 'inference_target_tts_text.txt'
    sample_size = 100
    min_text_length = 140
    max_text_length = 200

    # 加载所有文本行，去掉前缀
    all_lines = load_lines(combined_file, remove_prefix=True)

    # 加载已经被挑选出来的 1000 条语句（包含前缀）
    sampled_lines_with_prefix = load_lines(filtered_sampled_file, remove_prefix=False)
    sampled_texts = {text for _, text in sampled_lines_with_prefix}

    # 移除已经在 filtered_sampled_output.txt 中的语句
    remaining_lines = [line for line in all_lines if line not in sampled_texts]

    # 根据长度过滤剩余的文本
    filtered_remaining_lines = filter_by_length([(None, line) for line in remaining_lines], min_text_length, max_text_length)

    # 随机选择 1000 条满足条件的文本
    if len(filtered_remaining_lines) >= sample_size:
        new_sampled_lines = random.sample(filtered_remaining_lines, sample_size)
    else:
        print(f"Warning: Only {len(filtered_remaining_lines)} lines available after filtering.")
        new_sampled_lines = filtered_remaining_lines

    # 将选中的文本写入新的文件，只保留文本部分
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for _, text in new_sampled_lines:
            outfile.write(f"{text}\n")

    print(f"Newly sampled lines have been written to {output_file}")

if __name__ == "__main__":
    main()