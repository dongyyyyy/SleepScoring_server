# from include.header import *
# from models.transformer.attention import *
# from models.transformer.FeedForward import *

# class EncoderLayer(nn.Module):
#     def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
#         super().__init__()

#         self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
#         self.ff_layer_norm = nn.LayerNorm(hidden_dim)
#         self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
#         self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
#         self.dropout = nn.Dropout(dropout_ratio)

#     # 하나의 임베딩이 복제되어 Query, Key, Value로 입력되는 방식
#     def forward(self, src, src_mask):

#         # src: [batch_size, src_len, hidden_dim]
#         # src_mask: [batch_size, src_len]

#         # self attention
#         # 필요한 경우 마스크(mask) 행렬을 이용하여 어텐션(attention)할 단어를 조절 가능
#         _src, _ = self.self_attention(src, src, src, src_mask)

#         # dropout, residual connection and layer norm
#         src = self.self_attn_layer_norm(src + self.dropout(_src))

#         # src: [batch_size, src_len, hidden_dim]

#         # position-wise feedforward
#         _src = self.positionwise_feedforward(src)

#         # dropout, residual and layer norm
#         src = self.ff_layer_norm(src + self.dropout(_src))

#         # src: [batch_size, src_len, hidden_dim]

#         return src


# class Encoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length=100):
#         super().__init__()
#         self.device = device
#         self.tok_embedding = nn.Embedding(input_dim, hidden_dim)
#         self.pos_embedding = nn.Embedding(max_length, hidden_dim)

#         self.layers = nn.ModuleList([EncoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])

#         self.dropout = nn.Dropout(dropout_ratio)

#         self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

#     def forward(self, src, src_mask):

#         # src: [batch_size, src_len]
#         # src_mask: [batch_size, src_len]

#         batch_size = src.shape[0]
#         src_len = src.shape[1]

#         pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

#         # pos: [batch_size, src_len]

#         # 소스 문장의 임베딩과 위치 임베딩을 더한 것을 사용
#         src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

#         # src: [batch_size, src_len, hidden_dim]

#         # 모든 인코더 레이어를 차례대로 거치면서 순전파(forward) 수행
#         for layer in self.layers:
#             src = layer(src, src_mask)

#         # src: [batch_size, src_len, hidden_dim]

#         return src # 마지막 레이어의 출력을 반환