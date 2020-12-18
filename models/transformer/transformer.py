# from include.header import *
# from models.transformer.Encoder import *
# from models.transformer.Decoder import *
# from models.cnn.ResNet import *

# class Transformer(nn.Module):
#     def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
#         super().__init__()

#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_pad_idx = src_pad_idx
#         self.trg_pad_idx = trg_pad_idx
#         self.device = device

#     # 소스 문장의 <pad> 토큰에 대하여 마스크(mask) 값을 0으로 설정
#     def make_src_mask(self, src):

#         # src: [batch_size, src_len]

#         src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

#         # src_mask: [batch_size, 1, 1, src_len]

#         return src_mask

#     # 타겟 문장에서 각 단어는 다음 단어가 무엇인지 알 수 없도록(이전 단어만 보도록) 만들기 위해 마스크를 사용
#     def make_trg_mask(self, trg):

#         # trg: [batch_size, trg_len]

#         """ (마스크 예시)
#         1 0 0 0 0
#         1 1 0 0 0
#         1 1 1 0 0
#         1 1 1 0 0
#         1 1 1 0 0
#         """
#         trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

#         # trg_pad_mask: [batch_size, 1, 1, trg_len]

#         trg_len = trg.shape[1]

#         """ (마스크 예시)
#         1 0 0 0 0
#         1 1 0 0 0
#         1 1 1 0 0
#         1 1 1 1 0
#         1 1 1 1 1
#         """
#         trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()

#         # trg_sub_mask: [trg_len, trg_len]

#         trg_mask = trg_pad_mask & trg_sub_mask

#         # trg_mask: [batch_size, 1, trg_len, trg_len]

#         return trg_mask

#     def forward(self, src, trg):

#         # src: [batch_size, src_len]
#         # trg: [batch_size, trg_len]

#         src_mask = self.make_src_mask(src)
#         trg_mask = self.make_trg_mask(trg)

#         # src_mask: [batch_size, 1, 1, src_len]
#         # trg_mask: [batch_size, 1, trg_len, trg_len]

#         enc_src = self.encoder(src, src_mask)

#         # enc_src: [batch_size, src_len, hidden_dim]

#         output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

#         # output: [batch_size, trg_len, output_dim]
#         # attention: [batch_size, n_heads, trg_len, src_len]

#         return output, attention

# class DETR_psg(nn.Module):
#     def __init__(self, num_classes, hidden_dim=256, nheads=8,
#                  num_encoder_layers=6, num_decoder_layers=6,sequence_length=10):
#         super().__init__()

#         # create conversion layer
#         self.featureExtract = resnet18_200hz_withDropout_ensemble_branch_new_FE()

#         self.embed = nn.Linear(512*3, hidden_dim)
#         self.hidden_dim = hidden_dim
#         self.sequence_length = sequence_length
#         # create a default PyTorch transformer
#         self.transformer = nn.Transformer(
#             hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

#         # prediction heads, one extra class for predicting non-empty slots
#         # note that in baseline DETR linear_bbox layer is 3-layer MLP
#         self.linear_class = nn.Linear(hidden_dim, num_classes)

#         # output positional encodings (object queries)
#         self.query_pos = nn.Parameter(torch.rand(10, hidden_dim))

#         # spatial positional encodings
#         # note that in baseline DETR we use sine positional encodings
#         # [ 50 , hidden_dim//2 ]
#         self.row_embed = nn.Parameter(torch.rand(5, hidden_dim // 2))
#         self.col_embed = nn.Parameter(torch.rand(5, hidden_dim // 2))
#         print('row & col embed shape : ',self.row_embed.shape, self.col_embed.shape)
#     def forward(self, inputs):
#         # propagate inputs through ResNet-50 up to avg-pool layer
#         # print('input shape : ',inputs.shape)
#         x = self.featureExtract(inputs)
#         # print('feature shape : ',x.shape)
#         # convert from 2048 to 256 feature planes for the transformer
#         # (batch, hidden_dim)
#         h = self.embed(x)

#         src, mask = x[-1].decompose()

#         print('src shape : ',src.shape, 'mask shape : ',mask.shape)
#         # print('h shape : ', h.shape)
#         # (batch // sequence length , sequence length , hidden_dim)
#         h = h.reshape(-1,self.sequence_length,self.hidden_dim)
#         # construct positional encodings
#         # H = sequence length / W = hidden_dim
#         # H, W = h.shape[-2:]
#         # print(f'H shape : {H} / W shape : {W}')
#         pos = torch.cat([
#             self.col_embed.unsqueeze(0).repeat(2, 1, 1), # H, hidden_dim, hidden_dim//2
#             self.row_embed.unsqueeze(0).repeat(2, 1, 1), # H, hidden_dim, hidden_dim//2
#         ], dim=-1).flatten(0, 1).unsqueeze(1)
#         print(self.col_embed.unsqueeze(0).repeat(1, 1, 1).shape)
#         print(self.col_embed.unsqueeze(0).repeat(5, 1, 1).shape)
#         # print('self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1)',self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1).shape)
#         # print('self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)',self.row_embed[:H].unsqueeze(1).repeat(1, W, 1).shape)
#         # # [hidden_dim * ]
#         # print('pos shape : ',pos.shape)
#         # propagate through the transformer
#         print('pos shape : ',pos.shape,'shape : ',h.shape, ' self.query_pos.unsqueeze(1)).transpose(0, 1) shape : ',self.query_pos.unsqueeze(1).shape)
#         print('pos + 0.1 * h shape : ',(pos + 0.1 * h).shape, ' query shape : ',self.query_pos.unsqueeze(1).shape)
#         h = self.transformer(pos + 0.1 * h,
#                              self.query_pos.unsqueeze(1))
        
#         # finally project transformer outputs to class labels and bounding boxes
#         return self.linear_class(h)