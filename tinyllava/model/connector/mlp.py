import re

import torch.nn as nn

from . import register_connector
from .base import Connector
from .SparseAlign1 import GCSAligner


ACT_TYPE = {
    'relu': nn.ReLU,
    'gelu': nn.GELU
}

class MLPConnector(Connector):
    def __init__(self, config):
        super().__init__()
        
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', config.connector_type)
        act_type = config.connector_type.split('_')[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.vision_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            
        self.connector = nn.Sequential(*modules)
        self.align = GCSAligner(input_dim=896, proj_dim=896, k_intra=96)
    def forward(self, x, cur_input_embeds):

        image_features = self.connector(x)
        if cur_input_embeds==None:
            return image_features
        out = self.align(image_features[0], cur_input_embeds, use_sinkhorn=False, topk_align=256, coarse_candidate=None)
        Hx = out['Hx']
        Key_id = out['key_idx']
        out_image = Hx[Key_id]
        out_image = out_image.to(image_features.dtype)
        image_features = out_image.unsqueeze(0).repeat(image_features.shape[0], 1, 1)

        return image_features

@register_connector('mlp')
class MoFMLPConnector(Connector):
    def __init__(self, config):
        super().__init__()

        self._connector = MLPConnector(config)

   
        
#     @property
#     def config(self):
#         return {"connector_type": 'mlp',
#                 "in_hidden_size": self.in_hidden_size, 
#                 "out_hidden_size": self.out_hidden_size
#                }
    
