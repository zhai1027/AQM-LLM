import pandas as pd

# 读取原始 CSV 文件
file_path = r"C:\Users\59302\Desktop\SIT724 w12\w12_throughput.csv"
df = pd.read_csv(file_path)

# 将 throughput 从 Bytes/s 转换为 Bits/s
df['throughput_bits'] = df['throughput'] * 8

# 保存结果到新的 CSV 文件
output_path = r"C:\Users\59302\Desktop\SIT724 w12\SIT724_w12_throughput_bits.csv"
df.to_csv(output_path, index=False)

# 打印前几行结果以确认
print(df[['frame.time_epoch', 'throughput', 'throughput_bits']].head())
