# from include.header import *

# class MultiHeadAttentionLayer(nn.Module):
#     def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
#         super().__init__()

#         assert hidden_dim % n_heads == 0

#         self.hidden_dim = hidden_dim # Signals -> FeatureExtract를 통해 추출한 임베딩 차원
#         self.n_heads = n_heads # 헤드(head)의 개수: 서로 다른 어텐션(attention) 컨셉의 수
#         self.head_dim = hidden_dim // n_heads # 각 헤드(head)에서의 임베딩 차원

#         self.fc_q = nn.Linear(hidden_dim, hidden_dim) # Query 값에 적용될 FC 레이어
#         self.fc_k = nn.Linear(hidden_dim, hidden_dim) # Key 값에 적용될 FC 레이어
#         self.fc_v = nn.Linear(hidden_dim, hidden_dim) # Value 값에 적용될 FC 레이어

#         self.fc_o = nn.Linear(hidden_dim, hidden_dim)

#         self.dropout = nn.Dropout(dropout_ratio)

#         self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

#     def forward(self, query, key, value, mask = None): 

#         batch_size = query.shape[0] # batch size

#         # query: [batch_size, query_len, hidden_dim]
#         # key: [batch_size, key_len, hidden_dim]
#         # value: [batch_size, value_len, hidden_dim]
 
#         Q = self.fc_q(query) 
#         K = self.fc_k(key)
#         V = self.fc_v(value)

#         # Q: [batch_size, query_len, hidden_dim]
#         # K: [batch_size, key_len, hidden_dim]
#         # V: [batch_size, value_len, hidden_dim]

#         # hidden_dim → n_heads X head_dim 형태로 변형
#         # n_heads(h)개의 서로 다른 어텐션(attention) 컨셉을 학습하도록 유도
#         Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

#         # Q: [batch_size, n_heads, query_len, head_dim]
#         # K: [batch_size, n_heads, key_len, head_dim]
#         # V: [batch_size, n_heads, value_len, head_dim]

#         # Attention Energy 계산
#         energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

#         # energy: [batch_size, n_heads, query_len, key_len]

#         # 마스크(mask)를 사용하는 경우
#         if mask is not None:
#             # 마스크(mask) 값이 0인 부분을 -1e10으로 채우기
#             energy = energy.masked_fill(mask==0, -1e10)

#         # 어텐션(attention) 스코어 계산: 각 단어에 대한 확률 값
#         attention = torch.softmax(energy, dim=-1)

#         # attention: [batch_size, n_heads, query_len, key_len]

#         # 여기에서 Scaled Dot-Product Attention을 계산
#         x = torch.matmul(self.dropout(attention), V)

#         # x: [batch_size, n_heads, query_len, head_dim]

#         x = x.permute(0, 2, 1, 3).contiguous()

#         # x: [batch_size, query_len, n_heads, head_dim]

#         x = x.view(batch_size, -1, self.hidden_dim)

#         # x: [batch_size, query_len, hidden_dim]

#         x = self.fc_o(x)

#         # x: [batch_size, query_len, hidden_dim]

#         return x, attention