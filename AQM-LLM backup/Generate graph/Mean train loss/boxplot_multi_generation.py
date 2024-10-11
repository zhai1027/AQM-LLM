import pandas as pd
import matplotlib.pyplot as plt

# 文件路径列表
file_paths = [
    r"C:\Users\59302\Desktop\SIT724 w12\生成图像准备\2000.csv",
    r"C:\Users\59302\Desktop\SIT724 w12\生成图像准备\4000.csv",
    r"C:\Users\59302\Desktop\SIT724 w12\生成图像准备\8000.csv",
    r"C:\Users\59302\Desktop\SIT724 w12\生成图像准备\16000.csv"
]

# 图例标签列表
labels = ['2000 samples', '4000 samples', '8000 samples', '16000 samples']

# 收集所有 Mean Train Loss 数据
mean_train_loss_data = []

# 为每个文件收集 Mean Train Loss 数据
for file_path in file_paths:
    df = pd.read_csv(file_path)
    mean_train_loss_data.append(df['Mean Train Loss'])

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制箱线图
plt.boxplot(mean_train_loss_data, labels=labels)

# 添加标题和标签
plt.title('Comparison of Mean Train Loss Distribution')
plt.xlabel('Sample Sizes')
plt.ylabel('Mean Train Loss')

# 显示图表
plt.grid(True)
plt.show()
