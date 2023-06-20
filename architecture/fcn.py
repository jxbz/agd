import math
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, depth, width, input_dim, output_dim, bias=False):
        super(FCN, self).__init__()
        
        self.initial = nn.Linear(input_dim, width, bias=bias)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=bias) for _ in range(depth-2)])
        self.final = nn.Linear(width, output_dim, bias=bias)
        
    def forward(self, x):        
        x = x.view(x.shape[0],-1)

        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)
        
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x) * math.sqrt(2)
        
        return self.final(x)
