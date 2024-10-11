import pandas as pd

# 读取 CSV 文件
file_path = r"C:\Users\59302\Desktop\SIT724 w12\SIT724_data.csv"
df = pd.read_csv(file_path)

# 删除没有 RTT 值的行
df_filtered = df.dropna(subset=['tcp.analysis.ack_rtt'])

# 计算相邻数据包的时间差
df_filtered['time_diff'] = df_filtered['frame.time_epoch'].diff()

# 处理时间差为 0 或 NaN 的情况，将其替换为 NaN
df_filtered['time_diff'] = df_filtered['time_diff'].replace(0, float('nan'))

# 计算吞吐量（Bytes/s），跳过时间差为 NaN 或 0 的情况
df_filtered['throughput'] = df_filtered['frame.len'] / df_filtered['time_diff']

# 保存计算后的结果到新的 CSV 文件
output_path = r"C:\Users\59302\Desktop\SIT724 w12\w12_throughput.csv"
df_filtered.to_csv(output_path, index=False)

print(df_filtered[['frame.time_epoch', 'frame.len', 'time_diff', 'throughput']].head())

print(f"结果保存在地址： {output_path}")
