import torch
import torch.nn as nn

class PromptLinearNet(nn.Module):
    def __init__(self, hidden_dim, threshold=0.1) -> None:
        super().__init__()

        self.predictor = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 1),
                                       nn.Sigmoid())
        self.threshold = threshold

    def forward(self, ego_nodes, central_nodes):
        pred_logits = self.predictor(torch.cat([ego_nodes, central_nodes], dim=1))
        return pred_logits

    def make_prediction(self, ego_nodes, central_nodes):
        pred_logits = self.predictor(torch.cat([ego_nodes, central_nodes], dim=1)).squeeze(1)

        pos = torch.where(pred_logits >= self.threshold, 1.0, 0.0)
        return pos.nonzero().t().squeeze(0).detach().cpu().numpy().tolist()
