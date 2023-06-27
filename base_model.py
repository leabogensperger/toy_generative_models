from torch import nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def fit(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)