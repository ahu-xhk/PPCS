import torch
from torch import nn

from .layers import Swish, make_linear_block


class Agent(nn.Module):

    def __init__(self,
                 hidden_size,

                 norm_type=None
                 ):
        super().__init__()
        self.hidden_size = hidden_size 

        self.seed_embedding = nn.Linear(1, hidden_size, bias=False) 
        self.node_embedding = nn.Linear(1, hidden_size, bias=False)
        self.input_mapping = nn.Sequential( 
            make_linear_block(hidden_size, hidden_size, Swish, norm_type),
            make_linear_block(hidden_size, hidden_size, Swish, norm_type),
        )
        self.node_score_layer = nn.Linear(hidden_size, 1, bias=False) 
        self.stopping_score_layer = nn.Linear(hidden_size, 2, bias=False) 

  
        nn.init.zeros_(self.node_score_layer.weight.data)
        nn.init.zeros_(self.stopping_score_layer.weight.data)

    def forward(self, x_seeds, x_nodes, indptr):
        h = self.seed_embedding(x_seeds.unsqueeze(1)) + self.node_embedding(x_nodes.unsqueeze(1))
        h = self.input_mapping(h)
        node_scores = self.node_score_layer(h).squeeze(1)


        batch = []
        for startpoint, endpoint, candidate_endpoint in indptr:
            if startpoint == endpoint:
                raise ValueError('Finished Episode!')
            else:
                stop_node = torch.mean(h[startpoint:endpoint], dim=0).unsqueeze(0)
       
                node_logits = torch.log_softmax(node_scores[startpoint:candidate_endpoint], 0)  
                stopping_logits = torch.log_softmax(self.stopping_score_layer(stop_node), 1).squeeze(0) 
                logits = torch.cat([node_logits + stopping_logits[0], stopping_logits[1:]], dim=0) 
                batch.append(logits)         
        batch_logits = batch

        return batch_logits