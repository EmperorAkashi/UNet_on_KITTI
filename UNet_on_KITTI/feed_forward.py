import torch
import torch.nn as nn

class feed_forward:
    def __init__(self, in_ch, out_ch, hidden_layer):
        super(feed_forward).__init__()
        self.l1 = nn.Linear(in_ch, hidden_layer)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer, out_ch)
    def forward(self,x) -> torch.Tensor:
        output = self.l1(x) 
        output = self.relu(output)
        output = self.l2(output)
        return output

