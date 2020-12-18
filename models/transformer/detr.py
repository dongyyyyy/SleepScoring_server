# import torch
# import torch.nn.functional as F
# from torch import nn
# from models.cnn.ResNet import *
# from util import box_ops
# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized)

# from .backbone import build_backbone
# from .matcher import build_matcher
# from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
#                            dice_loss, sigmoid_focal_loss)
# from .transformer import build_transformer

# class DETR(nn.Module):
#     """ This is the DETR module that performs object detection """
#     def __init__(self, transformer, num_classes, num_queries, aux_loss=False):
#         super().__init__()
#         self.num_queries = num_queries
#         self.transformer = transformer
#         hidden_dim = transformer.d_model
#         self.class_embed = nn.Linear(hidden_dim, num_classes )
#         # self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
#         self.query_embed = nn.Embedding(num_queries, hidden_dim)
#         self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
#         self.backbone = resnet18_200hz_withDropout_ensemble_branch_new_FE()
#         self.aux_loss = aux_loss

#     def forward(self, x):
#         features, pos = self.backbone(x)

#         src, mask = features[-1].decompose()
#         assert mask is not None
#         hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

#         outputs_class = self.class_embed(hs)
#         # outputs_coord = self.bbox_embed(hs).sigmoid()
        
#         out = outputs_class[-1]

#         return out


