import pandas as pd
import matplotlib.pyplot as plt

# 文件路径列表
file_paths = [
    r"C:\Users\59302\Desktop\SIT724 w12\生成图像准备\Train loss std\2000.csv",
    r"C:\Users\59302\Desktop\SIT724 w12\生成图像准备\Train loss std\4000.csv",
    r"C:\Users\59302\Desktop\SIT724 w12\生成图像准备\Train loss std\8000.csv",
    r"C:\Users\59302\Desktop\SIT724 w12\生成图像准备\Train loss std\16000.csv"
]

# 图例标签列表
labels = ['2000 samples', '4000 samples', '8000 samples', '16000 samples']

# 创建一个空列表存储每个样本的 Train Loss Std 数据
train_loss_std_data = []

# 为每个文件读取 'Train Loss Std' 列并添加到列表中
for file_path in file_paths:
    df = pd.read_csv(file_path)
    train_loss_std_data.append(df['Train Loss Std'])

# 创建一个 box plot
plt.figure(figsize=(10, 6))
plt.boxplot(train_loss_std_data, labels=labels)

# 添加标题和标签
plt.title('Box Plot of Train Loss Std')
plt.xlabel('Samples')
plt.ylabel('Train Loss Std')

# 显示图表
plt.grid(True)
plt.show()
