import os

def read_normalized_files(root_dir, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # 遍历根目录及其子目录
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.normalized.txt'):
                    file_path = os.path.join(subdir, file)
                    base_name = os.path.splitext(file)[0].rsplit('.', 1)[0]  # 去掉 .normalized.txt
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()  # 读取并去除首尾空白字符
                        
                        # 写入输出文件，格式为 "文件名 内容"
                        output_file.write(f"{base_name} {content}\n")

if __name__ == "__main__":
    root_directory = '.'  # 当前目录
    output_file = 'combined_output.txt'  # 输出文件名
    
    read_normalized_files(root_directory, output_file)
    print(f"All .normalized.txt contents have been written to {output_file}")