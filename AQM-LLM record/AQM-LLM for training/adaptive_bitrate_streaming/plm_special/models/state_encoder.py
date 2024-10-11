import torch.nn as nn
import torch

class EncoderNetwork(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Linear layers for numerical features
        self.rtt_fc = nn.Linear(1, embed_dim)  # For RTT
        self.throughput_fc = nn.Linear(1, embed_dim)  # For Throughput

        # Embedding for ECN
        self.ecn_embedding = nn.Embedding(4, embed_dim)  # ECN can be 0, 1, 2, 3
        
        # Final fully connected layer to combine all features
        self.fc_final = nn.Linear(embed_dim * 3, embed_dim)  # 3 inputs: RTT, ECN, Throughput

    def forward(self, state):
        # Extract the components from the state
        rtt = state[..., 0].unsqueeze(-1)  # RTT
        ecn = state[..., 1].long()  # ECN
        throughput = state[..., 2].unsqueeze(-1)  # Throughput

        # 对 ECN 值进行强制处理，将不在 [0, 1, 3] 范围内的值默认设置为 0
        ecn = torch.where((ecn != 0) & (ecn != 1) & (ecn != 3), torch.tensor(0).to(ecn.device), ecn)

        # Process numerical features
        rtt_encoded = torch.relu(self.rtt_fc(rtt))
        throughput_encoded = torch.relu(self.throughput_fc(throughput))

        # Process ECN
        ecn_encoded = self.ecn_embedding(ecn)

        # Concatenate all encoded features
        combined = torch.cat([rtt_encoded, throughput_encoded, ecn_encoded], dim=-1)

        # Final output
        output = torch.relu(self.fc_final(combined))

        return output

