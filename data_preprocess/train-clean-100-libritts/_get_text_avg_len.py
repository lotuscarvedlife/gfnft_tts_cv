input_file_path = 'filtered_sampled_output.txt'
with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    avg_len = []
    for line in lines:
        avg_len.append(len(line.split(" ", 1)[1]))

    print(sum(avg_len)/len(avg_len))