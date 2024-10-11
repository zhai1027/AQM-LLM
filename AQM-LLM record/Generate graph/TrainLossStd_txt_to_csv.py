import re
import csv

log_file = r"C:\Users\59302\Desktop\SIT724 w12\生成图像准备\16000.txt"
output_csv = r"C:\Users\59302\Desktop\SIT724 w12\16000_train_loss_std.csv"

# 正则表达式模式，匹配 'training/train_loss_mean' 和 'training/train_loss_std' 后面的数字
train_loss_std_pattern = r"'training/train_loss_std': ([\d.]+)"  # 匹配 'training/train_loss_std' 后面的数字

# 初始化空列表用于存储结果
train_loss_stds = []

with open(log_file, 'r', encoding='utf-8') as f:
    content = f.read()

    # 查找所有符合模式的 'train loss std'
    train_loss_stds = re.findall(train_loss_std_pattern, content)

# 确保提取到的是正确的数量
print(f"提取到 {len(train_loss_stds)} 个 'train loss std' 值")

# 将提取的 'train loss std' 写入 CSV 文件
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    # 写入表头
    csvwriter.writerow(['Iteration', 'Train Loss Std'])

    # 写入每个 iteration 的 train loss std
    for i, loss_std in enumerate(train_loss_stds):
        csvwriter.writerow([i, loss_std])

print(f"'Train loss std' 结果已成功保存到 {output_csv}")
