import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class NormLinear(nn.Module):
    def __init__(self, in_features, out_features, temperature = 0.05, temperature_trainable = False):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.scale = 1 / temperature
        if temperature_trainable:
            self.scale = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.scale, 1 / temperature)

    def forward(self, x):
        x_norm = F.normalize(x)
        w_norm = F.normalize(self.weight)
        cosine = F.linear(x_norm, w_norm, None)
        out = cosine #* self.scale
        return out

class CosineMarginCrossEntropy(nn.Module):

    def __init__(self, m=0.60, s=30.0):
        super(CosineMarginCrossEntropy, self).__init__()
        self.m = m
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (input - one_hot * self.m)
        
        loss = self.ce(output, target)
        return loss
    
class ArcMarginCrossEntropy(nn.Module):

    def __init__(self, m=0.50, s=30.0, m_cos = 0.3):
        super(ArcMarginCrossEntropy, self).__init__()
        self.m = m
        self.m_cos = m_cos
        self.s = s
        self.ce = nn.CrossEntropyLoss() 
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        

    def forward(self, cosine, target):
        
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        #output = output - one_hot * self.m_cos # cosine-margin
        output *= self.s
        
        loss = self.ce(output, target)
        return loss     
