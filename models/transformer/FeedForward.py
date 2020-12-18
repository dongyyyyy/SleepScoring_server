# from include.header import *

# class PositionwiseFeedforwardLayer(nn.Module):
#     def __init__(self, hidden_dim, pf_dim, dropout_ratio):
#         super().__init__()

#         self.fc_1 = nn.Linear(hidden_dim, pf_dim)
#         self.fc_2 = nn.Linear(pf_dim, hidden_dim)

#         self.dropout = nn.Dropout(dropout_ratio)

#     def forward(self, x):

#         # x: [batch_size, seq_len, hidden_dim]
#         x = self.fc_1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         # x: [batch_size, seq_len, pf_dim]
        
#         x = self.fc_2(x)
#         # x: [batch_size, seq_len, hidden_dim]

#         return x