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

# 创建图形
plt.figure(figsize=(10, 6))

# 为每个文件绘制一条曲线
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    plt.plot(df['Iteration'], df['Mean Train Loss'], marker='o', linestyle='-', label=labels[i])

# 强制将 X 轴设置为 1 到 20 的整数刻度
plt.xticks(range(0, 20, 1))

# 添加标题和标签
plt.title('Comparison of Mean Train Loss')
plt.xlabel('Iteration')
plt.ylabel('Mean Train Loss')

# 添加图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()
