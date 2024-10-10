import argparse
import pandas as pd
import numpy as np
import os
import pickle
import torch

class ExperiencePool:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

def calculate_reward(rtt, throughput):
    if rtt <= 0 or throughput <= 0:
        return 0.1 

    performance_ratio = throughput / rtt

    reward = np.log10(performance_ratio)

    return reward

def run(args):
    df = pd.read_csv(args.csv_file)
    exp_pool = ExperiencePool()

    for index, row in df.iterrows():
        rtt = float(row['tcp.analysis.ack_rtt']) if pd.notna(row['tcp.analysis.ack_rtt']) else 0
        throughput = float(row['throughput']) if pd.notna(row['throughput']) else 0
        ecn_value = int(row['ip.dsfield.ecn']) if pd.notna(row['ip.dsfield.ecn']) else 0

        # 更新state为三个变量：RTT, ECN, Throughput
        """
        state = [
            rtt,            # RTT
            ecn_value,      # ECN标记
            throughput      # 吞吐量
        ]
        """
        # 将 state 转换为 Tensor
        state = torch.tensor([
            rtt,            # RTT
            ecn_value,      # ECN标记
            throughput      # 吞吐量
        ], dtype=torch.float32)
        
        # 定义action（基于ECN值）
        action = 1 if ecn_value == 1 else 3 if ecn_value == 3 else 0

        # 计算reward
        reward = calculate_reward(rtt, throughput)
        done = index == len(df) - 1

        # 将state, action, reward, done加入经验池
        exp_pool.add(state, action, reward, done)

    # 直接保存ExperiencePool实例
    exp_pool_path = os.path.join(args.output_dir, 'exp_pool.pkl')
    
    with open(exp_pool_path, 'wb') as f:
        pickle.dump(exp_pool, f)

    print(f"经验池已保存至: {exp_pool_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='根据L4S实验数据生成经验池')
    parser.add_argument('--csv_file', type=str, default=r"C:\Users\59302\Desktop\SIT724 w12\一万六千个样本.csv", help='CSV文件路径')
    parser.add_argument('--output_dir', type=str, default=r"C:\Users\59302\Desktop\SIT724 w12", help='保存经验池的目录')

    args = parser.parse_args()
    run(args)