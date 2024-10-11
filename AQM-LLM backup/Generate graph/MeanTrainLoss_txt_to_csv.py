import re
import csv

log_file = r"C:\Users\59302\Desktop\SIT724 w12\16000.txt"
output_csv = r"C:\Users\59302\Desktop\SIT724 w12\16000.csv"


train_loss_pattern = r"'training/train_loss_mean': ([\d.]+)" 

# 初始化空列表用于存储结果
train_loss_means = []

with open(log_file, 'r', encoding='utf-8') as f:
    content = f.read()

    # 查找所有符合模式的 'mean train loss'
    train_loss_means = re.findall(train_loss_pattern, content)

# 确保提取到的是正确的数量
print(f"提取到 {len(train_loss_means)} 个 'mean train loss' 值")

# 将提取的 'mean train loss' 写入 CSV 文件
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    # 写入表头
    csvwriter.writerow(['Iteration', 'Mean Train Loss'])

    # 写入每个 iteration 的 mean train loss
    for i, loss in enumerate(train_loss_means):
        csvwriter.writerow([i, loss])

print(f"'Mean train loss' 结果已成功保存到 {output_csv}")

