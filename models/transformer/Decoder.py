# from include.header import *
# from models.transformer.attention import *
# from models.transformer.FeedForward import *

# class DecoderLayer(nn.Module):
#     def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
#         super().__init__()

#         self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
#         self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim)
#         self.ff_layer_norm = nn.LayerNorm(hidden_dim)
#         self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
#         self.encoder_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
#         self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
#         self.dropout = nn.Dropout(dropout_ratio)

#     # 인코더의 출력 값(enc_src)을 어텐션(attention)하는 구조
#     def forward(self, trg, enc_src, trg_mask, src_mask):

#         # trg: [batch_size, trg_len, hidden_dim]
#         # enc_src: [batch_size, src_len, hidden_dim]
#         # trg_mask: [batch_size, trg_len]
#         # src_mask: [batch_size, src_len]

#         # self attention
#         # 자기 자신에 대하여 어텐션(attention)
#         _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

#         # dropout, residual connection and layer norm
#         trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

#         # trg: [batch_size, trg_len, hidden_dim]

#         # encoder attention
#         # 디코더의 쿼리(Query)를 이용해 인코더를 어텐션(attention)
#         _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

#         # dropout, residual connection and layer norm
#         trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

#         # trg: [batch_size, trg_len, hidden_dim]

#         # positionwise feedforward
#         _trg = self.positionwise_feedforward(trg)

#         # dropout, residual and layer norm
#         trg = self.ff_layer_norm(trg + self.dropout(_trg))

#         # trg: [batch_size, trg_len, hidden_dim]
#         # attention: [batch_size, n_heads, trg_len, src_len]

#         return trg, attention

# class Decoder(nn.Module):
#     def __init__(self, output_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length=100):
#         super().__init__()

#         self.device = device

#         self.tok_embedding = nn.Embedding(output_dim, hidden_dim)
#         self.pos_embedding = nn.Embedding(max_length, hidden_dim)

#         self.layers = nn.ModuleList([DecoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])

#         self.fc_out = nn.Linear(hidden_dim, output_dim)

#         self.dropout = nn.Dropout(dropout_ratio)

#         self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

#     def forward(self, trg, enc_src, trg_mask, src_mask):

#         # trg: [batch_size, trg_len]
#         # enc_src: [batch_size, src_len, hidden_dim]
#         # trg_mask: [batch_size, trg_len]
#         # src_mask: [batch_size, src_len]

#         batch_size = trg.shape[0]
#         trg_len = trg.shape[1]

#         pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

#         # pos: [batch_size, trg_len]

#         trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

#         # trg: [batch_size, trg_len, hidden_dim]

#         for layer in self.layers:
#             # 소스 마스크와 타겟 마스크 모두 사용
#             trg, attention = layer(trg, enc_src, trg_mask, src_mask)

#         # trg: [batch_size, trg_len, hidden_dim]
#         # attention: [batch_size, n_heads, trg_len, src_len]

#         output = self.fc_out(trg)

#         # output: [batch_size, trg_len, output_dim]

#         return output, attention