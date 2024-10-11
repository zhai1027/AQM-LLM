import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
file_path = r"C:\Users\59302\Desktop\SIT724 w12\生成图像准备\16000.csv"
df = pd.read_csv(file_path)

# 创建图形，使用 Iteration 作为 X 轴，Mean Train Loss 作为 Y 轴
plt.figure(figsize=(10, 6))
plt.plot(df['Iteration'], df['Mean Train Loss'], marker='o', linestyle='-', color='b')

# 强制将 X 轴设置为 1 到 20 的整数刻度
plt.xticks(range(0, 20, 1))

# 添加标题和标签
plt.title('16000 samples')
plt.xlabel('Iteration')
plt.ylabel('Mean Train Loss')

# 显示图表
plt.grid(True)
plt.show()