import pickle

# 定义 ExperiencePool 类
class ExperiencePool:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

# 指定PKL文件的路径
#pkl_file_path = r'C:\Users\59302\Desktop\SIT724 w12\2000个样本的经验池\exp_pool.pkl'
#pkl_file_path = r'C:\Users\59302\Desktop\SIT724 w12\4000个样本的经验池\exp_pool.pkl'
#pkl_file_path = r'C:\Users\59302\Desktop\SIT724 w12\6000个样本的经验池\exp_pool.pkl'
#pkl_file_path = r'C:\Users\59302\Desktop\SIT724 w12\8000个样本的经验池\exp_pool.pkl'
pkl_file_path = r'C:\Users\59302\Desktop\SIT724 w12\16000个样本的经验池\exp_pool.pkl'

# 加载 .pkl 文件
with open(pkl_file_path, 'rb') as f:
    exp_pool = pickle.load(f)

# 确认数据的结构
states = exp_pool.states
actions = exp_pool.actions
rewards = exp_pool.rewards
dones = exp_pool.dones

# 获取实际的样本数量
num_samples_available = len(states)
print(f"Total available samples: {num_samples_available}")

# 选择要提取的样本数量，确保不会超过实际样本总数
num_samples = min(214839, num_samples_available)

# 指定输出文件的路径
output_file_path = r'C:\Users\59302\Desktop\SIT724 w12\16000个样本的经验池\output_samples.txt'

# 打开输出文件，写入内容
with open(output_file_path, 'w') as output_file:
    # 打印样本的详细内容并写入文件
    for i in range(num_samples):
        output_file.write(f"State: {states[i]}\n")
        output_file.write(f"Action: {actions[i]}\n")
        output_file.write(f"Reward: {rewards[i]}\n")
        output_file.write(f"Done: {dones[i]}\n")
        output_file.write(" " * 10 + "\n")

print(f"Samples successfully saved to {output_file_path}")
