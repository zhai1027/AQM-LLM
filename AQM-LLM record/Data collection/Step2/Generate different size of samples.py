import pandas as pd

# 读取CSV文件
csv_file = r"C:\Users\59302\Desktop\SIT724 w12\SIT724_w12_3个经验池变量.csv"
df = pd.read_csv(csv_file)

# 确保读取的数据没有丢失列，特别是'tcp.analysis.ack_rtt'
print("数据列名:", df.columns)

# 定义截取范围的起始和结束索引
first_ecn_position = 17043
last_ecn_position = 17229
front_end_value = 7907

start_index = first_ecn_position - front_end_value
end_index = last_ecn_position + front_end_value

# 确保截取数据时不会丢失 'RTT' 等重要列
selected_rows = df.iloc[start_index:end_index + 1]

# 检查是否有空值并打印出可能的异常行
print("是否有空值:", selected_rows.isnull().sum())

# 保存数据到新的CSV文件
output_file = r"C:\Users\59302\Desktop\SIT724 w12\一万六千个样本.csv"
selected_rows.to_csv(output_file, index=False)

print(f"新数据集已生成，包含 {len(selected_rows)} 行。")

