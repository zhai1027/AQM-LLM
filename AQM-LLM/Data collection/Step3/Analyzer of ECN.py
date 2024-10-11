import pandas as pd

# 读取CSV文件
csv_file = r"C:\Users\59302\Desktop\SIT724 w12\四千个样本.csv"
df = pd.read_csv(csv_file)

# 统计 ip.dsfield.ecn 等于 1 和 3 的行数
ecn_0_count = (df['ip.dsfield.ecn'] == 0).sum()
ecn_1_count = (df['ip.dsfield.ecn'] == 1).sum()
ecn_3_count = (df['ip.dsfield.ecn'] == 3).sum()

# 打印统计结果
print(f"ECN=0 的数量: {ecn_0_count}")
print(f"ECN=1 的数量: {ecn_1_count}")
print(f"ECN=3 的数量: {ecn_3_count}")

