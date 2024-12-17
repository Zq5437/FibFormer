import subprocess
import os

# 定义不同的 ratio 值
# ratios = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
ratios = [0.8]

# 结果保存路径前缀
result_path_prefix = "/home/h666/Zq/2025ICASSP/baseline/CA-MSER/ratio_results"
os.makedirs(result_path_prefix, exist_ok=True)

# 遍历每个 ratio 值
for ratio in ratios:
    # 构建命令
    command = f"python crossval_SER.py --ratio {ratio}"
    
    # 构建输出文件名
    output_file = os.path.join(result_path_prefix, f"output_ratio_{ratio}.txt")
    
    print(f"运行命令: {command}, 输出保存为: {output_file}")
    
    # 运行命令并将输出保存到文件
    with open(output_file, 'w') as f:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b''):
            f.write(line.decode('utf-8'))
            f.flush()  # 确保缓冲区内容被写入文件
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            print(f"命令 {command} 执行失败，返回代码: {return_code}")

print("所有任务已完成。")
